# FMN Machine Risk Monitor

A monitoring tool that flags which machines on the plant floor are at risk of failure soon, explains why in plain English, and lets a plant manager ask free text questions about the current state of the fleet.

Live app: https://fmn-machine-risk-monitor-u3vumcph2doegcnyw4vuee.streamlit.app/

## 1. Problem Understanding

FMN wanted a way to catch machine failures before they happen, using sensor data that is already being collected, and to make that information usable by someone on the plant floor who is not a data person.

The data behind this covers 17 machines over four months (January to April 2026), with hourly readings for vibration and temperature, about 43,000 rows in total. Out of all that data, only 17 real failure events happened, about one per machine on average. That is far too few examples to train a normal supervised classifier that can be trusted. A model that simply always predicted "no failure" would already be right 99.96% of the time and would be completely useless.

So instead of building a classifier, I treated this as a risk scoring problem. I combined a few signals that are known to move before a failure into a single score, and used that score to flag machines as Low, Medium, or High risk.

The brief also asked for two specific AI features on top of this:
- A plain English explanation for any flagged machine, generated at the moment it is viewed, grounded in that machine's real numbers, not a generic sentence.
- A free text question box where someone can ask something like "which machines need attention this week" and get an answer generated from the actual current data.

## 2. Approach

The system is split into two layers that are kept strictly separate, and that separation is the most important design decision in the whole project: **the risk engine decides, the AI only explains.**

### High-level flow

Raw sensor data goes through cleaning and feature engineering, which works out each machine's own normal range for vibration and temperature. That feeds into the risk scoring step, which combines several signals into one score and sorts each machine into Low, Medium, or High risk. That risk flag is checked against the known failures to make sure the approach actually works. The result then feeds a Streamlit dashboard, where Gemini explains each flagged machine and answers free text questions about the fleet.

### Data preparation and feature engineering

Handled in `app/data_prep.py`, built out and tested in `notebook/01_data_prep.ipynb`. This step cleans the raw sensor data, fills short gaps, and calculates each machine's own normal range for vibration and temperature, meaning its own mean and standard deviation, not a fleet-wide average. The 48 hours before any of that machine's own past failures are excluded from this calculation, so a machine's own incidents do not distort its own definition of normal.

### Risk scoring

Also in `app/data_prep.py`, built out and tested in `notebook/02_model_and_risk_flags.ipynb`. Four signals are combined into a single score between 0 and roughly 1:
- Vibration deviation from that machine's own normal (the strongest and cleanest signal)
- Temperature deviation from that machine's own normal (a strong second signal, in fact slightly more elevated on average at real failures than vibration was)
- Whether vibration has been climbing over the last 24 hours
- Hours since the last maintenance (a small extra risk multiplier, not a strong signal on its own, since actual failures in the data ranged from 15 hours to almost 3,000 hours since service with no clean pattern)

That score is then split into Low, Medium, and High using two thresholds. All the weights and thresholds live in `risk_config.json`, not hard coded in the app, so they can be retuned without touching code.

### Why this approach

A traditional classifier needs many examples of both outcomes to learn from. With only 17 failures across four months, that path was not realistic and would have produced a model that looked accurate on paper while never actually being useful. A transparent risk score built from known warning signals, and checked directly against the real failures that happened, is more honest about what the data can actually support, and easier for a plant manager to trust because every flag can be traced back to a specific number.

### Validation

The thresholds were checked against naive single-signal baselines, a sweep of possible thresholds against the 24 to 48 hour warning window the brief asked for, and a leave-one-failure-out test, where each of the 17 known failures was held out in turn and the system was checked to see if it still caught it using thresholds tuned on the other 16.

| Check | Result |
|---|---|
| Leave-one-failure-out recall | ~94% |
| Precision on distinct alert episodes | ~66% |

This is described honestly as retrospective calibration rather than independent validation, since the threshold was tuned using the same 17 failures it is being judged against. The leave-one-out check exists specifically to guard against that being pure overfitting.

### LLM explanation

Handled in `app/explain.py`, using Google's Gemini model. Once the risk engine has already decided a machine's status, this layer takes that machine's real numbers, current vibration and temperature, how far each is from that machine's own normal (in plain words like "well above normal" rather than a raw statistic), the 24 hour trend, and hours since maintenance, and asks Gemini to explain it in a few natural sentences. The model is explicitly instructed never to change the risk level, never to invent a mechanical cause, and never to state a probability of failure. It only explains a decision that has already been made elsewhere. The wording is deliberately varied between calls so it does not read like a filled in template.

### Grounded fleet Q&A

The free text question box passes the current status of all 17 machines into a single prompt and lets Gemini answer directly from that, with the same rule that it cannot invent machines, numbers, or causes that are not in the data provided.

### The dashboard

Built with Streamlit in `app/app.py`. It shows one large number up top, how many machines need attention right now, a queue of exactly those machines with a one line plain language reason for each, a full sortable table of all 17 machines color coded by risk, and a detail view for any machine that opens with its AI explanation first, followed by the actual readings and a 7 day trend chart. The most urgent machine opens automatically when the app loads, nobody has to click anything to see what needs attention.

One honest note on the data shown by default: the very last hour in the dataset happens to be an unusually calm one, since it does not line up with any of the 17 real failures. Rather than showing that quiet hour and making the tool look like it never finds anything, the dashboard is pinned to a real hour from earlier in the dataset, February 23, 2026 at 6am, where three machines are genuinely flagged at once (two High, one Medium). This is not invented data, it is a real moment from the dataset, and it is a strong one: the machine flagged High that morning (MCH-206) failed for real 16 hours later, and two more machines flagged that same morning (MCH-204 and MCH-205) both failed for real four days later. In a live deployment reading from real sensors, this would simply be whatever hour it currently is.

### Project structure

fmn-machine-risk-monitor/
│
├── app/
│ ├── app.py
│ ├── data_prep.py
│ ├── explain.py
│ └── requirements.txt
│
├── notebook/
│ ├── 01_data_prep.ipynb
│ └── 02_model_and_risk_flags.ipynb
│
├── data_prepared.csv
├── data_scored.csv
├── machine_baselines.csv
├── project2_manufacturing_sensors.csv
└── risk_config.json


## 3. How to Run

### Run it locally

Clone the repository:
git clone https://github.com/Imoleoluwanimi/fmn-machine-risk-monitor.git
cd fmn-machine-risk-monitor

Install the dependencies:
pip install -r app/requirements.txt

Add your Gemini API key. Create a file at `app/.streamlit/secrets.toml` containing:
GEMINI_API_KEY = "your-key-here"
This file is intentionally excluded from the repository through `.gitignore` and must be created locally, or set as a secret if deploying.

Run the app:
streamlit run app/app.py
The app expects `risk_config.json` and `data_scored.csv` to sit at the repository root, one level above the `app` folder, which is already how this repository is laid out.

### Dependencies

Listed in `app/requirements.txt`: Streamlit, pandas, numpy, plotly, and requests (used to call the Gemini API directly through its OpenAI compatible endpoint).

### Deployed version

The app is deployed on Streamlit Community Cloud at the link at the top of this file. The Gemini API key is stored there as a Streamlit secret rather than in the repository.

## 4. Limitations & Next Steps

**What is incomplete or simplified:**
- The 94% recall figure comes from retrospective calibration on only 17 failures. That is a small sample, and the thresholds have not been tested against genuinely new, unseen failures. More real failure data over time would make this number much more trustworthy.
- The dashboard's "current" view is pinned to a specific real hour in the historical dataset rather than a live sensor feed, since this project works from a fixed CSV rather than a real time data source.
- Uploading a new CSV through the sidebar replaces the entire fleet dataset, it does not append new readings to the existing 17 machines or let you add a new machine on its own.
- There is no login or user management, anyone with the link can view the dashboard and use the Q&A box.
- The maintenance hours signal is currently a fairly blunt multiplier. The data did not show a clean relationship between hours since maintenance and failure severity, so it is weighted low on purpose, but a larger dataset might reveal a better way to use it.

**What would be improved with more time:**
1. Connect the pipeline to a real, live sensor feed instead of a static CSV, so "current" genuinely means right now.
2. Support appending new sensor readings for existing machines and registering new machines, rather than only replacing the whole dataset.
3. Add simple alerting, for example an email or message when a machine crosses into High risk, instead of requiring someone to open the dashboard.
4. Retrain and re-validate the risk thresholds as more real failures accumulate, moving away from a single retrospective calibration over time.
5. Add basic access control before this is used with real plant data outside of an evaluation setting.

## 5. Key Takeaway

This project combines a transparent, explainable risk scoring engine with generative AI used strictly for explanation, not decision-making:
- The risk engine decides, using signals that are traceable back to real sensor readings.
- Gemini explains that decision in plain language and answers questions grounded in the same data.

The key design principle is to keep the risk decision deterministic and checkable against real failures, and use the LLM only to make that decision easier for a plant manager to understand.
