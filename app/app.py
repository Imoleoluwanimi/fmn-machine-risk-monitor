"""
FMN Machine Risk Monitor
Project 2 of the FMN AI Engineering Internship technical assessment.

This file is the UI layer only. All risk scoring logic lives in
data_prep.py and all LLM explanation logic lives in explain.py.
This file must not reimplement or duplicate either.

Design notes for this pass:
- Base page is a cool, quiet off white with a faint sage undertone, not
  the generic warm cream look, and not a flat white void either. Cards
  sit on top of it with real elevation so the page has depth.
- FMN green is the working accent color, used on buttons, focus states,
  and the calm end of the headline panel. FMN yellow is used sparingly,
  as a single top accent stripe on the header and a thin accent under the
  headline number, never as a large surface.
- Risk severity (red, amber, green) is a separate palette from the two
  brand colors above, used only for status, and it drives the color of
  the headline panel itself so the single most important number on the
  page is impossible to miss.
- The machine detail view leads with the plain language explanation,
  supporting numbers and the trend chart come after, since the sentence
  is the thing a plant manager actually needs first.
- The most urgent machine is opened automatically on load. Nobody should
  have to click anything to see what needs attention.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_prep
import explain

# ---------------------------------------------------------------------------
# Brand colors (structural chrome only, never used for risk severity)
# ---------------------------------------------------------------------------
FMN_GREEN = "#18864B"
FMN_GREEN_DARK = "#0F5C33"
FMN_YELLOW = "#FFD500"
FMN_CHARCOAL = "#1A1A1A"

# Risk severity palette (status only, never used for brand or structure)
RISK_HIGH = "#C6423D"
RISK_HIGH_BG = "#FBEAE9"
RISK_MEDIUM = "#D98A1F"
RISK_MEDIUM_TEXT = "#9C5E0E"
RISK_MEDIUM_BG = "#FBF1E1"
RISK_LOW = "#2F9E5B"
RISK_LOW_BG = "#EAF6EF"
RISK_NEUTRAL = "#6B7280"

RISK_COLORS = {"High": RISK_HIGH, "Medium": RISK_MEDIUM, "Low": RISK_LOW}

# Base palette
BG_PAGE = "#EEF2EF"
BG_PANEL = "#FFFFFF"
BORDER_SOFT = "#DCE2DF"
TEXT_MUTED = "#5B6165"

# ---------------------------------------------------------------------------
# Demo snapshot
# ---------------------------------------------------------------------------
# The true latest hour in the bundled dataset (Apr 30, 2026, 11pm) has
# almost no risk activity, since the dataset's last recorded hour does not
# coincide with any of the 17 real failures. Rather than adding a
# time-travel control, the "current" view is fixed to a real hour that has
# genuine, simultaneous risk across the fleet: February 23, 2026, 6am.
# Three machines are flagged at once here (two High, one Medium), and this
# is real, validated data: MCH-206 goes on to fail for real 16 hours later,
# and MCH-204 and MCH-205 both fail for real four days later. This is not
# a fabricated scenario, it is a real early warning window from the
# dataset, chosen because it demonstrates the tool working, unlike the
# dataset's own final hour.
#
# If a freshly uploaded CSV does not cover this date, the app falls back
# to that file's own true latest hour automatically.
DEMO_SNAPSHOT_TIMESTAMP = pd.Timestamp("2026-02-23 06:00:00")

APP_DIR = Path(__file__).resolve().parent


def _find_project_file(filename):
    for candidate_dir in (APP_DIR, APP_DIR.parent):
        candidate = candidate_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in {APP_DIR} and {APP_DIR.parent}."
    )


CONFIG_PATH = _find_project_file("risk_config.json")
DATA_PATH = _find_project_file("data_scored.csv")

# NOTE: data_prep.py does not include deviation_label() or snapshot_as_of(),
# so both are implemented locally here rather than touching that file.


def deviation_label(zscore):
    """Plain word translation of a z-score, no statistics terminology."""
    if zscore is None or pd.isna(zscore):
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


def snapshot_as_of(df_scored, as_of_timestamp):
    """One row per machine, its most recent reading at or before the given
    timestamp."""
    df = df_scored[df_scored["timestamp"] <= as_of_timestamp]
    if df.empty:
        return df
    return df.sort_values("timestamp").groupby("machine_id", as_index=False).last()


def get_current_snapshot(df_scored):
    snap = snapshot_as_of(df_scored, DEMO_SNAPSHOT_TIMESTAMP)
    if snap.empty:
        return data_prep.latest_snapshot(df_scored), df_scored["timestamp"].max()
    return snap, DEMO_SNAPSHOT_TIMESTAMP


def build_reason(row):
    """The single most notable signal for a machine, in plain words."""
    vib_label = deviation_label(row["vibration_zscore"])
    temp_label = deviation_label(row["temperature_zscore"])
    trend = row["vib_trend_24h_pct"]

    candidates = [
        (abs(row["vibration_zscore"]), f"Vibration is {vib_label}."),
        (abs(row["temperature_zscore"]), f"Temperature is {temp_label}."),
    ]
    if pd.notna(trend) and abs(trend) >= 10:
        direction = "climbing" if trend > 0 else "easing"
        candidates.append(
            (abs(trend) / 10, f"Vibration is {direction}, {abs(trend):.0f}% over the last 24 hours.")
        )
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def highlight_risk_rows(row):
    status = row["Status"]
    if "High" in status:
        bg = "rgba(198, 66, 61, 0.10)"
    elif "Medium" in status:
        bg = "rgba(217, 138, 31, 0.12)"
    else:
        bg = "rgba(47, 158, 91, 0.06)"
    return [f"background-color: {bg};"] * len(row)


def style_status_text(status):
    if "High" in status:
        return f"color: {RISK_HIGH}; font-weight: 700;"
    if "Medium" in status:
        return f"color: {RISK_MEDIUM_TEXT}; font-weight: 700;"
    return f"color: {RISK_LOW}; font-weight: 600;"


# ---------------------------------------------------------------------------
# Page config and CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FMN Machine Risk Monitor",
    page_icon="\U0001F3ED",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .stApp {{
        background-color: {BG_PAGE};
    }}
    [data-testid="stSidebar"] {{
        background-color: #F7F9F7;
        border-right: 1px solid {BORDER_SOFT};
        border-top: 4px solid {FMN_YELLOW};
    }}

    .fmn-header {{
        background-color: {FMN_CHARCOAL};
        border-top: 5px solid {FMN_YELLOW};
        border-left: 6px solid {FMN_GREEN};
        padding: 1.4rem 1.7rem;
        border-radius: 10px;
        margin-bottom: 1.3rem;
    }}
    .fmn-header h1 {{
        color: white;
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    .fmn-header p {{
        color: #C9CCD1;
        margin: 0.35rem 0 0 0;
        font-size: 0.93rem;
    }}

    .hero-panel {{
        border-radius: 12px;
        border-left: 6px solid {FMN_GREEN};
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .headline-number {{
        font-size: 4.6rem;
        font-weight: 700;
        line-height: 1;
        margin: 0;
        letter-spacing: -0.02em;
        border-bottom: 5px solid {FMN_YELLOW};
        display: inline-block;
        padding-bottom: 0.1rem;
    }}
    .headline-label {{
        font-size: 1.15rem;
        font-weight: 500;
        color: {FMN_CHARCOAL};
        margin-top: 0.5rem;
    }}
    .chip-row {{
        margin-top: 0.9rem;
    }}
    .chip {{
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }}
    .snapshot-caption {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        margin-top: 0.7rem;
    }}

    .panel-card {{
        background-color: {BG_PANEL};
        border: 1px solid {BORDER_SOFT};
        border-left: 4px solid {FMN_GREEN};
        border-top: 2px solid {FMN_YELLOW};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}

    .risk-badge {{
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }}

    .queue-row {{
        border-bottom: 1px solid {BORDER_SOFT};
        padding: 0.75rem 0.1rem;
    }}
    .queue-row:last-child {{
        border-bottom: none;
    }}
    .queue-machine {{
        font-size: 1.02rem;
        font-weight: 700;
        color: {FMN_CHARCOAL};
    }}
    .queue-line {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
    }}

    .assistant-bubble {{
        border-radius: 12px;
        border-left: 4px solid {FMN_GREEN};
        background-color: rgba(24,134,75,0.06);
        padding: 1.1rem 1.3rem;
        font-size: 1.02rem;
        line-height: 1.55;
        color: #1F2421;
        margin: 0.6rem 0 1.1rem 0;
    }}

    .support-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {TEXT_MUTED};
        text-transform: none;
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }}

    .line-tag {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
    }}

    /* Make the Q&A text input actually visible against the page background */
    div[data-testid="stTextInput"] input {{
        background-color: {BG_PANEL} !important;
        border: 1px solid #B8C0BB !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border: 1px solid {FMN_GREEN} !important;
        box-shadow: 0 0 0 1px {FMN_GREEN} !important;
    }}

    .stButton > button {{
        border-radius: 7px;
        border: 1.5px solid {FMN_GREEN};
        color: {FMN_GREEN_DARK};
        background-color: white;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {FMN_GREEN};
        color: white;
        border-color: {FMN_GREEN};
    }}
    .stButton > button:focus {{
        border-color: {FMN_GREEN} !important;
        box-shadow: 0 0 0 1px {FMN_GREEN} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "selected_machine" not in st.session_state:
    st.session_state.selected_machine = None
if "explanation_cache" not in st.session_state:
    st.session_state.explanation_cache = {}


@st.cache_data
def load_config():
    return data_prep.load_config(str(CONFIG_PATH))


@st.cache_data
def load_scored_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


config = load_config()

if "scored_df" not in st.session_state:
    st.session_state.scored_df = load_scored_data()

df_scored = st.session_state.scored_df

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="fmn-header">
        <h1>FMN Machine Risk Monitor</h1>
        <p>Shows which machines need attention, based on how far each one's
        readings are running from its own normal pattern.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Current snapshot
# ---------------------------------------------------------------------------
snapshot, current_timestamp = get_current_snapshot(df_scored)

n_high = int((snapshot["risk_level"] == "High").sum())
n_medium = int((snapshot["risk_level"] == "Medium").sum())
n_total = int(len(snapshot))
n_limited = int(snapshot["limited_history"].sum())
n_attention = n_high + n_medium

attention_df = snapshot[snapshot["risk_level"].isin(["Medium", "High"])].copy()
attention_df = attention_df.sort_values("risk_score", ascending=False)

# ---------------------------------------------------------------------------
# Headline status, the single most important thing on the page
# ---------------------------------------------------------------------------
if n_high > 0:
    hero_text = RISK_HIGH
elif n_medium > 0:
    hero_text = RISK_MEDIUM_TEXT
else:
    hero_text = RISK_LOW

chip_html = (
    f"<span class='chip' style='background-color:{RISK_HIGH_BG}; color:{RISK_HIGH};'>{n_high} high risk</span>"
    f"<span class='chip' style='background-color:{RISK_MEDIUM_BG}; color:{RISK_MEDIUM_TEXT};'>{n_medium} medium risk</span>"
    f"<span class='chip' style='background-color:rgba(24,134,75,0.12); color:{FMN_GREEN_DARK};'>{n_total} machines monitored</span>"
)
if n_limited > 0:
    chip_html += (
        f"<span class='chip' style='background-color:rgba(255,213,0,0.28); color:{FMN_CHARCOAL};'>"
        f"{n_limited} still building a baseline</span>"
    )

st.markdown(
    f"""
    <div class="hero-panel" style="background-color:rgba(24,134,75,0.07);">
        <span class="headline-number" style="color:{hero_text};">{n_attention}</span>
        <div class="headline-label">machine{'s' if n_attention != 1 else ''} need attention</div>
        <div class="chip-row">{chip_html}</div>
        <div class="snapshot-caption">Fleet status as of {current_timestamp.strftime('%B %d, %Y, %I:%M %p')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Attention queue
# ---------------------------------------------------------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.subheader("Attention queue")

if attention_df.empty:
    st.success("No machines currently need attention. All monitored machines are within normal range.")
else:
    for _, row in attention_df.iterrows():
        badge_color = RISK_COLORS.get(row["risk_level"], RISK_NEUTRAL)
        reason = build_reason(row)
        cols = st.columns([2.4, 4.1, 1.3])
        with cols[0]:
            st.markdown(
                f"<div class='queue-machine'>{row['machine_id']}</div>"
                f"<div class='queue-line'>Production line: {row['line']}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f"<span class='risk-badge' style='background-color:{badge_color};'>"
                f"{row['final_status']}</span><br>{reason}",
                unsafe_allow_html=True,
            )
        with cols[2]:
            if st.button("View details", key=f"attn_{row['machine_id']}"):
                st.session_state.selected_machine = row["machine_id"]
        st.markdown("<div class='queue-row'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Default selection: the plant manager should never have to click to see
# the most urgent machine.
# ---------------------------------------------------------------------------
if st.session_state.selected_machine is None:
    if not attention_df.empty:
        st.session_state.selected_machine = attention_df.iloc[0]["machine_id"]
    elif not snapshot.empty:
        st.session_state.selected_machine = snapshot.sort_values(
            "risk_score", ascending=False
        ).iloc[0]["machine_id"]

# ---------------------------------------------------------------------------
# Machine detail, explanation first, numbers and chart after
# ---------------------------------------------------------------------------
if st.session_state.selected_machine:
    machine_id = st.session_state.selected_machine
    machine_row = snapshot[snapshot["machine_id"] == machine_id]

    if not machine_row.empty:
        row = machine_row.iloc[0]
        badge_color = RISK_COLORS.get(row["risk_level"], RISK_NEUTRAL)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        top_l, top_r = st.columns([5, 1])
        with top_l:
            st.markdown(
                f"### {row['machine_id']} "
                f"<span class='risk-badge' style='background-color:{badge_color};'>{row['final_status']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<span class='line-tag'>Production line: {row['line']}</span>", unsafe_allow_html=True)
        with top_r:
            refresh_clicked = st.button("Refresh", key=f"refresh_{machine_id}")

        cache_key = f"{machine_id}_{current_timestamp.isoformat()}"
        if cache_key not in st.session_state.explanation_cache or refresh_clicked:
            with st.spinner("Thinking..."):
                st.session_state.explanation_cache[cache_key] = explain.explain_machine_risk(row)

        st.markdown(
            f"<div class='assistant-bubble'>"
            f"{st.session_state.explanation_cache[cache_key]}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='support-label'>Readings behind this assessment</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        d1.metric("Vibration", f"{row['vibration_mm_s']:.2f} mm/s", deviation_label(row["vibration_zscore"]))
        d2.metric("Temperature", f"{row['temperature_c']:.1f} C", deviation_label(row["temperature_zscore"]))
        d3.metric("Hours since maintenance", f"{int(row['run_hours_since_maintenance'])}")

        trend = row["vib_trend_24h_pct"]
        if pd.notna(trend):
            trend_direction = "up" if trend > 0 else "down" if trend < 0 else "flat"
            st.caption(f"Vibration trend: {trend_direction} {abs(trend):.0f}% over the last 24 hours")
        else:
            st.caption("Not enough history yet to calculate a 24-hour trend.")

        machine_history = df_scored[
            (df_scored["machine_id"] == machine_id)
            & (df_scored["timestamp"] > current_timestamp - pd.Timedelta(days=7))
            & (df_scored["timestamp"] <= current_timestamp)
        ].sort_values("timestamp")

        if not machine_history.empty:
            baseline_mean = machine_history["vib_baseline_mean"].iloc[0]
            baseline_std = machine_history["vib_baseline_std"].iloc[0]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(machine_history["timestamp"]) + list(machine_history["timestamp"][::-1]),
                    y=[baseline_mean + baseline_std] * len(machine_history)
                    + [baseline_mean - baseline_std] * len(machine_history),
                    fill="toself",
                    fillcolor="rgba(24,134,75,0.12)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Normal range",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=machine_history["timestamp"],
                    y=machine_history["vibration_mm_s"],
                    mode="lines",
                    line=dict(color=FMN_CHARCOAL, width=2),
                    name="Vibration",
                )
            )
            fig.update_layout(
                title="Vibration, last 7 days",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=False,
                plot_bgcolor=BG_PANEL,
                paper_bgcolor=BG_PANEL,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Full fleet table
# ---------------------------------------------------------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.subheader("Full fleet")
st.caption("All monitored machines, worst first. Click a row to open it above.")

display_df = snapshot[
    ["machine_id", "line", "final_status", "risk_score", "temperature_c", "vibration_mm_s"]
].copy()
display_df = display_df.rename(
    columns={
        "machine_id": "Machine",
        "line": "Line",
        "final_status": "Status",
        "risk_score": "Risk score",
        "temperature_c": "Temp (C)",
        "vibration_mm_s": "Vibration (mm/s)",
    }
)
display_df["Risk score"] = display_df["Risk score"].round(2)
display_df = display_df.sort_values("Risk score", ascending=False).reset_index(drop=True)

styled_df = display_df.style.apply(highlight_risk_rows, axis=1).map(style_status_text, subset=["Status"])
styled_df = styled_df.set_table_styles(
    [{"selector": "thead th", "props": [("background-color", FMN_GREEN), ("color", "white")]}]
)

event = st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="fleet_table",
)

if event and event.selection and event.selection.rows:
    selected_row_idx = event.selection.rows[0]
    st.session_state.selected_machine = display_df.iloc[selected_row_idx]["Machine"]

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fleet Q&A
# ---------------------------------------------------------------------------
st.markdown('<div class="panel-card">', unsafe_allow_html=True)
st.subheader("Ask about your machines")
st.caption("Answers are generated live from current fleet data.")

example_questions = [
    "Which machines are most at risk right now?",
    "Are any machines on Line A showing rising vibration?",
    "Which machines haven't had maintenance in a long time?",
]

qa_cols = st.columns(len(example_questions))
example_clicked = None
for col, q in zip(qa_cols, example_questions):
    if col.button(q, key=f"example_{q}"):
        example_clicked = q

question = st.text_input("Type a question", value=example_clicked or "", label_visibility="collapsed", placeholder="Type a question about current fleet status")

if st.button("Ask") and question.strip():
    with st.spinner("Checking fleet status..."):
        answer = explain.answer_fleet_question(question, snapshot)
    st.markdown(answer)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### <span style='color:{FMN_GREEN};'>FMN</span> Machine Risk Monitor", unsafe_allow_html=True)

    st.markdown("**Upload new sensor data**")
    uploaded_file = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            df_raw = pd.read_csv(uploaded_file, parse_dates=["timestamp"])
            df_new_scored, _ = data_prep.run_full_pipeline(df_raw, config)
            st.session_state.scored_df = df_new_scored
            st.session_state.selected_machine = None
            st.session_state.explanation_cache = {}
        st.success("New data loaded.")
        st.rerun()

    st.divider()
    st.markdown("**What the risk levels mean**")
    st.markdown(
        f"""
        <span class='risk-badge' style='background-color:{RISK_LOW};'>Low</span>
        &nbsp; Within normal range<br><br>
        <span class='risk-badge' style='background-color:{RISK_MEDIUM};'>Medium</span>
        &nbsp; Worth keeping an eye on<br><br>
        <span class='risk-badge' style='background-color:{RISK_HIGH};'>High</span>
        &nbsp; Worth having someone check soon
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "This is a risk monitoring score based on each machine's own normal "
        "pattern, not a failure probability prediction."
    )
