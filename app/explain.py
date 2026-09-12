"""
Grounded LLM explanations and fleet Q&A for the FMN Machine Health
& Risk Monitor.

Important design decision:

The LLM does NOT calculate or decide machine risk.

Risk scores and Low / Medium / High classifications are calculated
deterministically by data_prep.py. The LLM receives those calculated
values and explains them in plain language for a maintenance team.

For the current fleet size of 17 machines, fleet Q&A uses the complete
latest machine snapshot as context instead of embeddings or vector
search. This keeps the implementation simple, transparent, and fully
grounded in the current fleet data.
"""

import random
import requests
import streamlit as st


GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)

GEMINI_MODEL = "gemini-3.6-flash"


EXPLANATION_STYLE_HINTS = [
    "Write like a quick note a technician would leave for the next shift.",
    "Write like you are briefing a supervisor who just walked up and asked what is going on.",
    "Write like a short maintenance-log entry, direct and matter-of-fact.",
    "Write like you are answering someone who just asked what is up with this machine.",
    "Write like a brief update at the start of a shift handover meeting.",
]


def _call_gemini(messages, max_tokens=2000):
    """
    Call Gemini through its OpenAI-compatible endpoint.
    """

    api_key = st.secrets["GEMINI_API_KEY"]

    response = requests.post(
        GEMINI_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GEMINI_MODEL,
            "messages": messages,
            "reasoning_effort": "low",
            "max_tokens": max_tokens,
            "temperature": 1.0,
        },
        timeout=60,
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def pd_isna(value):
    """
    Import-free NaN check.
    """

    try:
        return value != value
    except TypeError:
        return False


def deviation_label(zscore):
    """
    Plain-word translation of a z-score, no statistics terminology.

    This is the same wording used elsewhere in the app (data_prep does not
    expose this helper, so it is defined once here and once in app.py).
    Feeding the model this instead of a raw z-score keeps the prompt itself
    free of jargon, rather than showing the model a technical value and
    only telling it not to repeat that value back.
    """

    if zscore is None or pd_isna(zscore):
        return "not enough history yet"
    z = abs(zscore)
    if z < 1:
        return "typical for this machine"
    elif z < 2:
        return "a little above normal"
    elif z < 4:
        return "well above normal"
    else:
        return "far above normal"


def explain_machine_risk(machine_row):
    """
    Explain one machine's already-computed risk result.

    The model is explicitly prevented from:
    - deciding the risk level
    - inventing mechanical diagnoses
    - inventing failure probabilities
    - inventing numbers

    It can explain what the observed readings mean operationally and
    suggest a reasonable next step based only on those observations.
    """

    trend_pct = machine_row.get(
        "vib_trend_24h_pct"
    )

    if (
        trend_pct is not None
        and not pd_isna(trend_pct)
    ):
        trend_text = (
            f"{trend_pct:+.1f}% over the last 24 hours"
        )
    else:
        trend_text = (
            "not enough history yet to calculate a trend"
        )

    if machine_row.get("limited_history"):
        history_note = (
            "This machine has less than two weeks of history, "
            "so its normal range is still an early estimate."
        )
    else:
        history_note = (
            "This machine has enough history for its current "
            "baseline to be treated as established in this analysis."
        )

    style_hint = random.choice(
        EXPLANATION_STYLE_HINTS
    )

    prompt = f"""
You are a maintenance intelligence assistant for a flour mill. You are
talking directly to a plant manager who is not a data or engineering
person, through a chat style panel in a monitoring app.

You are NOT responsible for deciding whether this machine is Low,
Medium, or High risk. That decision has already been made by a
deterministic scoring system. Your job is to explain that existing
result, in your own natural words, using only the actual numbers below.

{style_hint}

Write like you are personally telling this plant manager what is going
on with this one machine and what you would do about it. Do not write
a memo. Do not use section headers, labels, or bullet points. Write two
or three short flowing sentences that first say what is happening and
why, then one short sentence with a practical next step.

Do not invent a mechanical fault or diagnosis. It is fine to suggest
checking the machine, watching the trend, or scheduling an inspection.
Do not claim that a bearing, motor, belt, shaft, or other component is
failing unless that is explicitly given to you.

Important rules:
- Do not change or reinterpret the risk level.
- Do not state a probability of failure.
- Do not invent numbers.
- Do not invent a mechanical cause.
- Do not claim the machine will fail.
- Do not say that a sensor reading proves a specific component is faulty.
- Do not use em dashes.
- Do not use section headers or bullet points, write flowing sentences.
- Do not mention that you are an AI or that this was generated by a model.
- Keep the entire answer to 3 or 4 sentences total.
- If risk is Low, say plainly that nothing unusual is showing and no
  action is needed, do not manufacture a concern.
- If the machine has limited history, mention in plain words that its
  normal range is still an early estimate, without using the word
  baseline.

Machine data:

Machine ID: {machine_row['machine_id']}
Production line: {machine_row['line']}

Risk level: {machine_row['risk_level']}
Risk score: {machine_row['risk_score']:.2f}

Current vibration:
{machine_row['vibration_mm_s']:.2f} mm/s

Normal vibration for this machine:
{machine_row['vib_baseline_mean']:.2f} mm/s

Typical vibration spread:
+/- {machine_row['vib_baseline_std']:.2f} mm/s

Vibration deviation from normal:
{deviation_label(machine_row['vibration_zscore'])}

24-hour vibration trend:
{trend_text}

Current temperature:
{machine_row['temperature_c']:.1f} C

Normal temperature for this machine:
{machine_row['temp_baseline_mean']:.1f} C

Typical temperature spread:
+/- {machine_row['temp_baseline_std']:.1f} C

Temperature deviation from normal:
{deviation_label(machine_row['temperature_zscore'])}

Hours since last maintenance:
{int(machine_row['run_hours_since_maintenance'])}

History context:
{history_note}
"""

    return _call_gemini(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )


def answer_fleet_question(
    question,
    snapshot_df,
):
    """
    Answer a maintenance question using the latest reading from
    every machine in the fleet.

    The fleet currently contains only 17 machines, so all current
    machine snapshots are placed directly into the model context.

    This is grounded context injection, not vector-search RAG.
    For a fleet this small, using the full snapshot avoids an
    unnecessary embedding/retrieval layer.
    """

    table_rows = []

    for _, row in snapshot_df.iterrows():

        trend = row.get(
            "vib_trend_24h_pct"
        )

        if (
            trend is not None
            and not pd_isna(trend)
        ):
            trend_string = f"{trend:+.1f}%"
        else:
            trend_string = "n/a"

        table_rows.append(
            f"""
Machine {row['machine_id']}
Production line: {row['line']}
Risk level: {row['risk_level']}
Risk score: {row['risk_score']:.2f}
Vibration: {row['vibration_mm_s']:.2f} mm/s
Vibration deviation: {deviation_label(row['vibration_zscore'])}
Temperature: {row['temperature_c']:.1f} C
Temperature deviation: {deviation_label(row['temperature_zscore'])}
24-hour vibration trend: {trend_string}
Hours since maintenance: {int(row['run_hours_since_maintenance'])}
Limited history: {bool(row['limited_history'])}
Latest reading: {row['timestamp']}
""".strip()
        )

    fleet_context = "\n\n".join(
        table_rows
    )

    prompt = f"""
You are a maintenance intelligence assistant for a flour mill, talking
directly to a plant manager who is not a data or engineering person.

Answer the user's question using ONLY the current machine data below.

The risk level and risk score have already been calculated by a
deterministic scoring system. Do not recalculate them and do not
invent additional machines, values, causes, or failure probabilities.

If the question cannot be answered from the supplied data, say that
plainly.

Write like you are personally answering a colleague, in plain
conversational language, not a report. Avoid section headers and
bullet points unless the user specifically asks for a list.

Do not use em dashes.

Keep the answer concise unless the user specifically asks for a list
or comparison.

Current fleet:

{fleet_context}

User question:

{question}
"""

    return _call_gemini(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )