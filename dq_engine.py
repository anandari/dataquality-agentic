import pandas as pd
import numpy as np
import yaml


def _load_rules(rules_path: str):
    with open(rules_path, "r") as f:
        return yaml.safe_load(f)["rules"]


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce")


def _effective_actual_pickup(row: dict):
    """If actual_pickup_ts is missing/NaT, fall back to planned_pickup_ts."""
    ap = row.get("actual_pickup_ts")
    if pd.isna(ap):
        ap = row.get("planned_pickup_ts")
    return ap


def _derive_fields(row: dict) -> dict:
    pp = row.get("planned_pickup_ts")
    ap = _effective_actual_pickup(row)
    pdv = row.get("planned_delivery_ts")
    adv = row.get("actual_delivery_ts")
    lu = row.get("last_update_ts")

    row["planned_transit_days"] = (pdv - pp).days if pd.notna(pdv) and pd.notna(pp) else np.nan
    row["actual_transit_days"] = (adv - ap).days if pd.notna(adv) and pd.notna(ap) else np.nan
    row["pickup_delay_days"] = (ap - pp).days if pd.notna(ap) and pd.notna(pp) else np.nan
    row["delivery_delay_days"] = (adv - pdv).days if pd.notna(adv) and pd.notna(pdv) else np.nan

    # Update delay: prefer last_update_ts, else fall back to milestone_update_lag_hours if present
    if pd.notna(lu) and pd.notna(adv):
        row["update_delay_hours"] = (lu - adv).total_seconds() / 3600.0
    elif pd.notna(row.get("milestone_update_lag_hours")):
        row["update_delay_hours"] = float(row.get("milestone_update_lag_hours"))
    else:
        row["update_delay_hours"] = np.nan

    return row


def _evaluate_condition(row: dict, condition: str) -> bool:
    if condition == "actual_delivery_ts_is_null":
        return pd.isna(row.get("actual_delivery_ts"))

    if condition == "carrier_is_null":
        v = row.get("carrier")
        return pd.isna(v) or str(v).strip() == ""

    if condition == "delivery_before_pickup":
        ad = row.get("actual_delivery_ts")
        ap = _effective_actual_pickup(row)
        return pd.notna(ad) and pd.notna(ap) and ad < ap

    if condition == "invalid_transit_time":
        at = row.get("actual_transit_days")
        return pd.notna(at) and (at < 0 or at > 60)

    if condition == "late_update":
        ud = row.get("update_delay_hours")
        return pd.notna(ud) and ud > 24

    return False


def score_single_shipment(record: dict, rules_path="dq_rules.yaml") -> dict:
    rules = _load_rules(rules_path)

    parsed = dict(record)
    for col in ["planned_pickup_ts", "actual_pickup_ts", "planned_delivery_ts", "actual_delivery_ts", "last_update_ts"]:
        parsed[col] = _to_dt(parsed.get(col))

    if "milestone_update_lag_hours" in parsed:
        parsed["milestone_update_lag_hours"] = pd.to_numeric(parsed.get("milestone_update_lag_hours"), errors="coerce")

    parsed = _derive_fields(parsed)

    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "NONE": 0}
    dq_score = 100
    max_severity = "NONE"
    dq_details = []
    dq_flags = []

    for rule in rules:
        name = rule["name"]
        cond = rule["condition"]
        penalty = int(rule["penalty"])
        severity = rule["severity"]

        if _evaluate_condition(parsed, cond):
            dq_score = max(dq_score - penalty, 0)
            dq_flags.append(name)
            dq_details.append({"rule": name, "severity": severity, "penalty": penalty, "condition": cond})
            if severity_rank[severity] > severity_rank[max_severity]:
                max_severity = severity

    return {
        "dq_score": dq_score,
        "max_severity": max_severity,
        "dq_details": dq_details,
        "dq_flags": ", ".join(dq_flags),
        "planned_transit_days": parsed.get("planned_transit_days"),
        "actual_transit_days": parsed.get("actual_transit_days"),
        "pickup_delay_days": parsed.get("pickup_delay_days"),
        "delivery_delay_days": parsed.get("delivery_delay_days"),
        "update_delay_hours": parsed.get("update_delay_hours"),
    }


def run_data_quality_checks(df: pd.DataFrame, rules_path="dq_rules.yaml") -> pd.DataFrame:
    df = df.copy()

    for col in ["planned_pickup_ts", "actual_pickup_ts", "planned_delivery_ts", "actual_delivery_ts", "last_update_ts"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "milestone_update_lag_hours" in df.columns:
        df["milestone_update_lag_hours"] = pd.to_numeric(df["milestone_update_lag_hours"], errors="coerce")

    df["planned_transit_days"] = (df["planned_delivery_ts"] - df["planned_pickup_ts"]).dt.days

    ap_eff = (
        df["actual_pickup_ts"].where(df["actual_pickup_ts"].notna(), df["planned_pickup_ts"])
        if "actual_pickup_ts" in df.columns
        else df["planned_pickup_ts"]
    )

    df["actual_transit_days"] = (df["actual_delivery_ts"] - ap_eff).dt.days
    df["pickup_delay_days"] = (ap_eff - df["planned_pickup_ts"]).dt.days
    df["delivery_delay_days"] = (df["actual_delivery_ts"] - df["planned_delivery_ts"]).dt.days

    if "last_update_ts" in df.columns:
        df["update_delay_hours"] = (df["last_update_ts"] - df["actual_delivery_ts"]).dt.total_seconds() / 3600
        if "milestone_update_lag_hours" in df.columns:
            df["update_delay_hours"] = df["update_delay_hours"].fillna(df["milestone_update_lag_hours"])
    elif "milestone_update_lag_hours" in df.columns:
        df["update_delay_hours"] = df["milestone_update_lag_hours"]
    else:
        df["update_delay_hours"] = np.nan

    df["dq_score"] = 100
    df["dq_flags"] = [[] for _ in range(len(df))]
    df["dq_details"] = [[] for _ in range(len(df))]
    df["max_severity"] = "NONE"

    rules = _load_rules(rules_path)
    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "NONE": 0}

    for idx, row in df.iterrows():
        row_dict = _derive_fields(row.to_dict())
        for rule in rules:
            name = rule["name"]
            cond = rule["condition"]
            penalty = int(rule["penalty"])
            severity = rule["severity"]

            if _evaluate_condition(row_dict, cond):
                df.at[idx, "dq_score"] = max(df.at[idx, "dq_score"] - penalty, 0)
                df.at[idx, "dq_flags"].append(name)
                df.at[idx, "dq_details"].append({"rule": name, "severity": severity, "penalty": penalty, "condition": cond})

                current = df.at[idx, "max_severity"]
                if severity_rank[severity] > severity_rank[current]:
                    df.at[idx, "max_severity"] = severity

    df["dq_flags"] = df["dq_flags"].apply(lambda x: ", ".join(x))

    # IQR outliers on delivery_delay_days
    s = pd.to_numeric(df["delivery_delay_days"], errors="coerce")
    if s.dropna().empty:
        df["delay_outlier"] = False
        return df

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    df["delay_outlier"] = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))

    return df
