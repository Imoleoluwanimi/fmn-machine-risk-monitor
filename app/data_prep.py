"""
Data cleaning, baseline calculation, and risk scoring for Project 2.

This mirrors exactly the logic developed and validated in
notebooks/01_data_prep.ipynb and notebooks/02_model_and_risk_flags.ipynb.
The weights and thresholds are not redecided here, they are loaded from
risk_config.json so the app always scores machines the same way the
notebooks validated, whether using the bundled dataset or a freshly
uploaded CSV in the same format.
"""

import json
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "machine_id", "line",
    "temperature_c", "vibration_mm_s",
    "run_hours_since_maintenance", "failure_event",
]

# A machine needs at least 2 weeks of hourly readings before its own mean
# and standard deviation are trusted the same way as an established
# machine. Matches FULL_HISTORY_MIN_HOURS in the notebook.
FULL_HISTORY_MIN_HOURS = 24 * 14

# How far back (in hours) a failure's own lead-up readings are excluded
# from that machine's baseline calculation, so a machine's incidents don't
# distort its own definition of "normal". Matches the notebook.
PREFAILURE_EXCLUSION_HOURS = 48


def load_config(config_path="../risk_config.json"):
    with open(config_path) as f:
        return json.load(f)


def validate_columns(df):
    """Check the uploaded file actually has the columns this pipeline needs,
    before doing anything else with it."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "This file is missing columns the pipeline needs: "
            + ", ".join(missing)
            + ". Expected the same format as project2_manufacturing_sensors.csv."
        )


def clean(df_raw):
    """Duplicate removal and short-gap interpolation, exactly as decided
    and justified in Notebook 1, Step 15."""
    validate_columns(df_raw)

    df = df_raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["machine_id", "timestamp"])
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    df["temperature_c"] = df.groupby("machine_id")["temperature_c"].transform(
        lambda s: s.interpolate(limit=2)
    )
    df["vibration_mm_s"] = df.groupby("machine_id")["vibration_mm_s"].transform(
        lambda s: s.interpolate(limit=2)
    )
    return df


def compute_baselines(df_clean):
    """Per-machine normal mean and standard deviation, excluding each
    machine's own pre-failure windows, exactly as justified in Notebook 1,
    Step 9, and reused for feature building in Notebook 1, Step 16."""
    df = df_clean.copy()
    fail_events = df[df["failure_event"] == 1][["machine_id", "timestamp"]]

    df["in_prefailure_window"] = False
    for _, frow in fail_events.iterrows():
        mask = (
            (df["machine_id"] == frow["machine_id"])
            & (df["timestamp"] > frow["timestamp"] - pd.Timedelta(hours=PREFAILURE_EXCLUSION_HOURS))
            & (df["timestamp"] <= frow["timestamp"])
        )
        df.loc[mask, "in_prefailure_window"] = True

    normal_df = df[~df["in_prefailure_window"]]
    baseline = normal_df.groupby("machine_id")[["temperature_c", "vibration_mm_s"]].agg(["mean", "std"])
    baseline.columns = ["temp_baseline_mean", "temp_baseline_std", "vib_baseline_mean", "vib_baseline_std"]
    return baseline.reset_index()


def build_features(df_clean, baseline_stats):
    """Vibration and temperature z-scores, the 24-hour vibration trend, and
    the limited-history flag, exactly as built in Notebook 1, Step 16."""
    df = df_clean.merge(baseline_stats, on="machine_id", how="left")

    df["vibration_zscore"] = (df["vibration_mm_s"] - df["vib_baseline_mean"]) / df["vib_baseline_std"]
    df["temperature_zscore"] = (df["temperature_c"] - df["temp_baseline_mean"]) / df["temp_baseline_std"]

    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    df["vib_roll6h"] = df.groupby("machine_id")["vibration_mm_s"].transform(
        lambda s: s.rolling(window=6, min_periods=3).mean()
    )
    df["vib_roll6h_24h_ago"] = df.groupby("machine_id")["vib_roll6h"].shift(24)
    df["vib_trend_24h_pct"] = (
        (df["vib_roll6h"] - df["vib_roll6h_24h_ago"]) / df["vib_roll6h_24h_ago"]
    ) * 100

    hours_per_machine = df.groupby("machine_id")["timestamp"].transform("count")
    df["limited_history"] = hours_per_machine < FULL_HISTORY_MIN_HOURS

    return df


def score(df_features, config):
    """Combine the features into the single risk score, and assign
    High / Medium / Low, exactly as built and validated in Notebook 2."""
    df = df_features.copy()
    z_cap = config["Z_CAP"]

    df["vib_component"] = df["vibration_zscore"].clip(lower=0, upper=z_cap) / z_cap
    df["temp_component"] = df["temperature_zscore"].clip(lower=0, upper=z_cap) / z_cap
    df["trend_component"] = df["vib_trend_24h_pct"].clip(lower=0, upper=100).fillna(0) / 100
    df["maintenance_component"] = (
        df["run_hours_since_maintenance"] / config["RUN_HOURS_SCALE"]
    ).clip(upper=1)

    df["risk_score"] = (
        config["W_VIBRATION"] * df["vib_component"]
        + config["W_TEMPERATURE"] * df["temp_component"]
        + config["W_TREND"] * df["trend_component"]
        + config["W_MAINTENANCE"] * df["maintenance_component"]
    )

    high_t = config["HIGH_RISK_THRESHOLD"]
    med_t = config["MEDIUM_RISK_THRESHOLD"]

    def level(s):
        if s >= high_t:
            return "High"
        elif s >= med_t:
            return "Medium"
        return "Low"

    df["risk_level"] = df["risk_score"].apply(level)
    df["final_status"] = np.where(
        df["limited_history"],
        "Early estimate (" + df["risk_level"] + ")",
        df["risk_level"],
    )
    return df


def run_full_pipeline(df_raw, config):
    """Clean, baseline, feature, and score a raw CSV in one call. Used both
    to build the bundled dataset and to rescore a freshly uploaded file."""
    df_clean = clean(df_raw)
    baseline_stats = compute_baselines(df_clean)
    df_features = build_features(df_clean, baseline_stats)
    df_scored = score(df_features, config)
    return df_scored, baseline_stats


def latest_snapshot(df_scored):
    """One row per machine, its most recent reading. This is what the
    dashboard shows, current status, not the full hourly history."""
    return (
        df_scored.sort_values("timestamp")
        .groupby("machine_id", as_index=False)
        .last()
    )
