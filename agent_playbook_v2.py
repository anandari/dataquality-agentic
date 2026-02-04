"""
Agent Playbook V2 - Adapted for processed_shipments_data.csv schema

Schema mapping:
  - ID -> shipment_id
  - Carrier -> carrier
  - Vessel -> vessel
  - POL -> origin (Port of Loading)
  - POD -> destination (Port of Discharge)
  - ETD -> planned_departure_ts (Estimated Time of Departure)
  - ATD -> actual_departure_ts (Actual Time of Departure)
  - ETA -> planned_arrival_ts (Estimated Time of Arrival)
  - ATA -> actual_arrival_ts (Actual Time of Arrival)
  - transit_days -> transit_days
  - ContainerCount, TotalWeight, TotalVolume, LegOrder -> operational fields
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import copy
import pandas as pd
import numpy as np

# --------------------------------------------------------
# PLAYBOOK: Rule definitions with meaning, owners, fixes, causes
# --------------------------------------------------------
PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "MISSING_CARRIER": {
        "meaning": "Carrier is missing; cannot analyze carrier performance or route optimization.",
        "owner": "Operations / Master Data",
        "fixes": [
            "Backfill from booking reference or contract data",
            "Infer from vessel name using carrier-vessel registry",
            "Mark as UNKNOWN and flag for manual review"
        ],
        "causes": [
            "Manual entry skipped field",
            "Carrier not assigned at booking",
            "Upstream integration missing carrier mapping"
        ],
    },
    "MISSING_VESSEL": {
        "meaning": "Vessel name is missing; limits ability to track voyage-level performance.",
        "owner": "Operations / Carrier Integration",
        "fixes": [
            "Query carrier API for vessel assignment",
            "Backfill from AIS data using port call times",
            "Accept as null for non-ocean modes (air/rail/truck)"
        ],
        "causes": [
            "Shipment is multimodal (air/rail segment)",
            "Vessel not yet assigned",
            "EDI message missing vessel field"
        ],
    },
    "MISSING_ACTUAL_ARRIVAL": {
        "meaning": "Actual arrival (ATA) is missing; delivery KPIs cannot be computed.",
        "owner": "Integration / Data Engineering",
        "fixes": [
            "Backfill from milestone feed if shipment is delivered",
            "Check if shipment is still in transit (expected behavior)",
            "Query carrier tracking API for latest status"
        ],
        "causes": [
            "Shipment still in transit",
            "Carrier milestone not received",
            "Integration delay"
        ],
    },
    "MISSING_ACTUAL_DEPARTURE": {
        "meaning": "Actual departure (ATD) is missing; transit time calculations may be unreliable.",
        "owner": "Integration / Carrier Data",
        "fixes": [
            "Use ETD as fallback if departure is confirmed",
            "Query port departure records",
            "Flag for manual verification"
        ],
        "causes": [
            "Departure milestone not transmitted",
            "Shipment cancelled or rerouted",
            "EDI integration gap"
        ],
    },
    "ARRIVAL_BEFORE_DEPARTURE": {
        "meaning": "Arrival timestamp is before departure - logically impossible.",
        "owner": "Data Engineering / Carrier Integration",
        "fixes": [
            "Check timezone normalization (UTC vs local)",
            "Validate against raw milestone feed",
            "Swap ATD and ATA if clearly reversed"
        ],
        "causes": [
            "Timezone conversion error",
            "Swapped timestamp fields",
            "Data entry error"
        ],
    },
    "NEGATIVE_TRANSIT_TIME": {
        "meaning": "Transit time is negative, indicating data quality issue.",
        "owner": "Data Quality / Ops Analytics",
        "fixes": [
            "Recalculate from corrected timestamps",
            "Apply absolute value if sign error",
            "Flag for manual review if magnitude is extreme"
        ],
        "causes": [
            "Timestamp ordering issue",
            "Calculation bug in upstream system",
            "Missing or null dates causing bad math"
        ],
    },
    "EXTREME_TRANSIT_TIME": {
        "meaning": "Transit time exceeds realistic bounds (>120 days or <0).",
        "owner": "Data Quality / Ops Analytics",
        "fixes": [
            "Verify against shipping lane typical transit",
            "Check for date parsing issues (year errors)",
            "Route to exception queue for manual review"
        ],
        "causes": [
            "Year parsing error (e.g., 2019 vs 2018)",
            "Wrong shipment leg matched",
            "Transshipment delays not accounted for"
        ],
    },
    "MISSING_PORT": {
        "meaning": "Port of Loading or Discharge is missing; routing analysis impossible.",
        "owner": "Master Data / Operations",
        "fixes": [
            "Infer from booking or contract data",
            "Use vessel schedule to derive ports",
            "Flag as incomplete booking"
        ],
        "causes": [
            "Incomplete booking data",
            "Multi-leg shipment with missing segment",
            "Upstream system integration gap"
        ],
    },
}

SEV_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

# --------------------------------------------------------
# DQ Rules for this schema
# --------------------------------------------------------
DQ_RULES = [
    {"name": "MISSING_CARRIER", "condition": "carrier_is_null", "penalty": 15, "severity": "MEDIUM"},
    {"name": "MISSING_VESSEL", "condition": "vessel_is_null", "penalty": 5, "severity": "LOW"},
    {"name": "MISSING_ACTUAL_ARRIVAL", "condition": "ata_is_null", "penalty": 20, "severity": "HIGH"},
    {"name": "MISSING_ACTUAL_DEPARTURE", "condition": "atd_is_null", "penalty": 10, "severity": "MEDIUM"},
    {"name": "ARRIVAL_BEFORE_DEPARTURE", "condition": "arrival_before_departure", "penalty": 30, "severity": "HIGH"},
    {"name": "NEGATIVE_TRANSIT_TIME", "condition": "negative_transit", "penalty": 25, "severity": "HIGH"},
    {"name": "EXTREME_TRANSIT_TIME", "condition": "extreme_transit", "penalty": 15, "severity": "MEDIUM"},
    {"name": "MISSING_PORT", "condition": "port_is_null", "penalty": 10, "severity": "MEDIUM"},
]


def _to_dt(x):
    """Convert to datetime, handling various formats."""
    if pd.isna(x):
        return pd.NaT
    return pd.to_datetime(x, errors="coerce")


def _evaluate_condition(row: dict, condition: str) -> bool:
    """Evaluate a DQ rule condition against a row."""
    if condition == "carrier_is_null":
        v = row.get("Carrier")
        return pd.isna(v) or str(v).strip() == ""

    if condition == "vessel_is_null":
        v = row.get("Vessel")
        return pd.isna(v) or str(v).strip() == ""

    if condition == "ata_is_null":
        return pd.isna(row.get("ATA"))

    if condition == "atd_is_null":
        return pd.isna(row.get("ATD"))

    if condition == "arrival_before_departure":
        ata = row.get("ATA")
        atd = row.get("ATD")
        if pd.notna(ata) and pd.notna(atd):
            return ata < atd
        return False

    if condition == "negative_transit":
        transit = row.get("transit_days")
        return pd.notna(transit) and transit < 0

    if condition == "extreme_transit":
        transit = row.get("transit_days")
        return pd.notna(transit) and (transit > 120 or transit < -30)

    if condition == "port_is_null":
        pol = row.get("POL")
        pod = row.get("POD")
        pol_null = pd.isna(pol) or str(pol).strip() == ""
        pod_null = pd.isna(pod) or str(pod).strip() == ""
        return pol_null or pod_null

    return False


def score_single_shipment(record: dict) -> dict:
    """Score a single shipment record against DQ rules."""
    parsed = dict(record)

    # Parse timestamps
    for col in ["ETD", "ATD", "ETA", "ATA"]:
        parsed[col] = _to_dt(parsed.get(col))

    # Ensure transit_days is numeric
    parsed["transit_days"] = pd.to_numeric(parsed.get("transit_days"), errors="coerce")

    severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "NONE": 0}
    dq_score = 100
    max_severity = "NONE"
    dq_details = []
    dq_flags = []

    for rule in DQ_RULES:
        name = rule["name"]
        cond = rule["condition"]
        penalty = int(rule["penalty"])
        severity = rule["severity"]

        if _evaluate_condition(parsed, cond):
            dq_score = max(dq_score - penalty, 0)
            dq_flags.append(name)
            dq_details.append({
                "rule": name,
                "severity": severity,
                "penalty": penalty,
                "condition": cond
            })
            if severity_rank[severity] > severity_rank[max_severity]:
                max_severity = severity

    # Calculate derived fields
    atd = parsed.get("ATD")
    ata = parsed.get("ATA")
    etd = parsed.get("ETD")
    eta = parsed.get("ETA")

    actual_transit = None
    if pd.notna(ata) and pd.notna(atd):
        actual_transit = (ata - atd).total_seconds() / 86400

    planned_transit = None
    if pd.notna(eta) and pd.notna(etd):
        planned_transit = (eta - etd).total_seconds() / 86400

    delay_days = None
    if pd.notna(ata) and pd.notna(eta):
        delay_days = (ata - eta).total_seconds() / 86400

    return {
        "dq_score": dq_score,
        "max_severity": max_severity,
        "dq_details": dq_details,
        "dq_flags": ", ".join(dq_flags),
        "actual_transit_days": actual_transit,
        "planned_transit_days": planned_transit,
        "delay_days": delay_days,
    }


def _dedupe(seq: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _observations_from_row(r: dict, scored: dict) -> List[str]:
    """Generate human-readable observations from a row."""
    obs = []
    if scored.get("dq_flags"):
        obs.append(f"Triggered rules: {scored.get('dq_flags')}")
    if r.get("transit_days") is not None:
        obs.append(f"transit_days = {r.get('transit_days')}")
    if scored.get("delay_days") is not None:
        obs.append(f"delay_days = {scored.get('delay_days'):.1f}")
    if r.get("Carrier"):
        obs.append(f"Carrier = {r.get('Carrier')}")
    if r.get("POL") and r.get("POD"):
        obs.append(f"Route = {r.get('POL')} -> {r.get('POD')}")
    return obs


def _confidence_from_evidence(dq_details: List[dict], severity: str) -> Tuple[float, List[str]]:
    """Calculate confidence score based on evidence."""
    reasons = []
    score = 0.35  # base

    if severity == "HIGH":
        score += 0.25
        reasons.append("HIGH severity rule triggered")
    elif severity == "MEDIUM":
        score += 0.15
        reasons.append("MEDIUM severity rule triggered")
    elif severity == "LOW":
        score += 0.05
        reasons.append("LOW severity only")

    if dq_details:
        score += min(0.25, 0.08 * len(dq_details))
        reasons.append(f"{len(dq_details)} rule(s) provide evidence")

    if len(dq_details) == 1 and severity in ["LOW", "NONE"]:
        score -= 0.10
        reasons.append("Only one weak signal")

    score = max(0.0, min(1.0, score))
    return score, reasons


# --------------------------------------------------------
# MULTI-CANDIDATE FIX GENERATOR
# --------------------------------------------------------
def _build_fix_candidates(row: dict, before: dict) -> List[dict]:
    """Generate candidate fixes based on triggered rules."""
    triggered = [d.get("rule") for d in before.get("dq_details", [])]
    candidates = []

    def change(from_val, to_val):
        return {"from": from_val, "to": to_val}

    # Fix 1: Swap ATD/ATA if ARRIVAL_BEFORE_DEPARTURE
    if "ARRIVAL_BEFORE_DEPARTURE" in triggered:
        def apply_swap(rec):
            rec2 = copy.deepcopy(rec)
            atd = rec2.get("ATD")
            ata = rec2.get("ATA")
            if pd.isna(atd) or pd.isna(ata):
                return rec2, {}
            rec2["ATD"], rec2["ATA"] = ata, atd
            # Recalculate transit
            if pd.notna(rec2["ATD"]) and pd.notna(rec2["ATA"]):
                rec2["transit_days"] = (rec2["ATA"] - rec2["ATD"]).total_seconds() / 86400
            return rec2, {
                "ATD": change(atd, rec2["ATD"]),
                "ATA": change(ata, rec2["ATA"]),
                "transit_days": change(rec.get("transit_days"), rec2["transit_days"]),
            }
        candidates.append({
            "name": "Swap ATD and ATA timestamps",
            "risk": "MEDIUM",
            "notes": "Common cause: swapped events or timestamp field mapping error.",
            "apply_fn": apply_swap
        })

    # Fix 2: Set missing carrier to UNKNOWN
    if "MISSING_CARRIER" in triggered:
        def apply_carrier_unknown(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("Carrier")
            rec2["Carrier"] = "UNKNOWN"
            return rec2, {"Carrier": change(old, "UNKNOWN")}
        candidates.append({
            "name": "Set missing Carrier to 'UNKNOWN'",
            "risk": "LOW",
            "notes": "Safe placeholder; should be enriched from booking data when available.",
            "apply_fn": apply_carrier_unknown
        })

    # Fix 3: Backfill ATA from ETA if missing
    if "MISSING_ACTUAL_ARRIVAL" in triggered:
        def apply_backfill_ata(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("ATA")
            eta = rec2.get("ETA")
            if pd.isna(eta):
                return rec2, {}
            rec2["ATA"] = eta
            return rec2, {
                "ATA": change(old, eta),
                "_note": "Backfilled ATA from ETA (assumes on-time arrival)"
            }
        candidates.append({
            "name": "Backfill ATA from ETA (proposal)",
            "risk": "HIGH",
            "notes": "Only valid if shipment is confirmed delivered. May mask actual delays.",
            "apply_fn": apply_backfill_ata
        })

    # Fix 4: Backfill ATD from ETD if missing
    if "MISSING_ACTUAL_DEPARTURE" in triggered:
        def apply_backfill_atd(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("ATD")
            etd = rec2.get("ETD")
            if pd.isna(etd):
                return rec2, {}
            rec2["ATD"] = etd
            return rec2, {
                "ATD": change(old, etd),
                "_note": "Backfilled ATD from ETD (assumes on-time departure)"
            }
        candidates.append({
            "name": "Backfill ATD from ETD (proposal)",
            "risk": "MEDIUM",
            "notes": "Reasonable if departure was confirmed but timestamp not captured.",
            "apply_fn": apply_backfill_atd
        })

    # Fix 5: Absolute value for negative transit
    if "NEGATIVE_TRANSIT_TIME" in triggered:
        def apply_abs_transit(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("transit_days")
            if pd.isna(old):
                return rec2, {}
            rec2["transit_days"] = abs(old)
            return rec2, {
                "transit_days": change(old, rec2["transit_days"]),
                "_note": "Applied absolute value to negative transit"
            }
        candidates.append({
            "name": "Apply absolute value to transit_days",
            "risk": "MEDIUM",
            "notes": "Quick fix for sign errors; root cause should still be investigated.",
            "apply_fn": apply_abs_transit
        })

    # Fix 6: Mark extreme transit for review
    if "EXTREME_TRANSIT_TIME" in triggered:
        def apply_cap_transit(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("transit_days")
            if pd.isna(old):
                return rec2, {}
            # Cap at reasonable bounds
            rec2["transit_days"] = max(0, min(120, old))
            return rec2, {
                "transit_days": change(old, rec2["transit_days"]),
                "_note": "Capped transit_days to 0-120 day range"
            }
        candidates.append({
            "name": "Cap transit_days to realistic bounds (0-120)",
            "risk": "HIGH",
            "notes": "Masks underlying data issue; use only for reporting if source cannot be fixed.",
            "apply_fn": apply_cap_transit
        })

    return candidates


def _simulate_fix_and_rescore(row: dict) -> dict:
    """
    Agent loop:
      - Score baseline
      - Generate fix candidates
      - Apply each candidate
      - Re-score each candidate
      - Pick best improvement
    """
    before = score_single_shipment(row)
    base_score = before["dq_score"]

    results = []
    candidates = _build_fix_candidates(row, before)

    for c in candidates:
        new_rec, proposed_changes = c["apply_fn"](row)
        after = score_single_shipment(new_rec)

        results.append({
            "fix_name": c["name"],
            "risk": c["risk"],
            "notes": c["notes"],
            "score_before": base_score,
            "score_after": after["dq_score"],
            "delta": after["dq_score"] - base_score,
            "proposed_changes": proposed_changes,
            "after_details": after["dq_details"],
        })

    # Choose best by (score_after, -risk, delta)
    risk_rank = {"LOW": 2, "MEDIUM": 1, "HIGH": 0}
    results_sorted = sorted(
        results,
        key=lambda r: (r["score_after"], risk_rank.get(r["risk"], 0), r["delta"]),
        reverse=True
    )

    best = {
        "attempted": len(results_sorted) > 0,
        "score_before": base_score,
        "score_after": base_score,
        "improved": False,
        "fix_name": None,
        "risk": None,
        "notes": None,
        "proposed_changes": {},
        "before_details": before["dq_details"],
        "after_details": before["dq_details"],
        "candidates_evaluated": results_sorted[:12],
        "note": "No fix attempted."
    }

    if results_sorted:
        top = results_sorted[0]
        best.update({
            "fix_name": top["fix_name"],
            "risk": top["risk"],
            "notes": top["notes"],
            "score_after": top["score_after"],
            "improved": top["score_after"] > base_score,
            "proposed_changes": top["proposed_changes"],
            "after_details": top["after_details"],
            "note": "Agent evaluated multiple fix candidates, re-scored each, and selected the best improvement."
        })

    return best


def triage_shipment(row: dict) -> dict:
    """
    Main triage function: analyze a shipment, score it, suggest fixes.

    Args:
        row: Dict with keys matching the processed_shipments_data.csv schema

    Returns:
        Triage result with summary, observations, recommendations, and fix simulation
    """
    shipment_id = row.get("ID", "UNKNOWN")
    carrier = row.get("Carrier", "NA")
    lane = f"{row.get('POL', 'NA')} -> {row.get('POD', 'NA')}"
    vessel = row.get("Vessel", "NA")

    # Score the shipment
    scored = score_single_shipment(row)

    dq_details: List[dict] = scored.get("dq_details", []) or []
    severity = scored.get("max_severity", "NONE")
    dq_score = scored.get("dq_score", None)

    dq_details_sorted = sorted(
        dq_details,
        key=lambda d: (SEV_RANK.get(d.get("severity", "NONE"), 0), d.get("penalty", 0)),
        reverse=True
    )

    top_rules = [d.get("rule") for d in dq_details_sorted if d.get("rule")]
    top_rules = top_rules[:5]

    observations = _observations_from_row(row, scored)

    likely_causes = []
    actions = []
    owners = set()

    for rname in top_rules:
        pb = PLAYBOOK.get(rname)
        if not pb:
            continue
        owners.add(pb["owner"])
        likely_causes.extend(pb["causes"][:2])
        actions.extend(pb["fixes"][:2])

    likely_causes = _dedupe(likely_causes)
    actions = _dedupe(actions)

    confidence_score, confidence_reasons = _confidence_from_evidence(dq_details_sorted, severity)
    confidence_label = "LOW"
    if confidence_score >= 0.72:
        confidence_label = "HIGH"
    elif confidence_score >= 0.50:
        confidence_label = "MEDIUM"

    fix_result = _simulate_fix_and_rescore(row)

    reasoning_trace = [
        f"Loaded evidence for shipment_id={shipment_id}, carrier={carrier}, lane={lane}",
        f"Observed dq_score={dq_score}, severity={severity}",
        f"Rules triggered (ordered) = {', '.join(top_rules) if top_rules else 'None'}",
        f"Confidence computed from severity + number of rules: {confidence_score:.2f} ({confidence_label})",
        f"Generated {len(fix_result.get('candidates_evaluated', []))} evaluated fix candidate(s).",
    ]

    if fix_result.get("attempted"):
        reasoning_trace.append("Tool-use loop: applied each candidate → re-scored → selected best.")
        reasoning_trace.append(f"Best candidate: {fix_result.get('fix_name')} (risk={fix_result.get('risk')})")
        reasoning_trace.append(f"Score before={fix_result.get('score_before')} → after={fix_result.get('score_after')}")
        if fix_result.get("improved"):
            reasoning_trace.append("Best candidate improved score. Recommend applying with approval and audit tag.")
        else:
            reasoning_trace.append("No candidate improved score. Recommend manual review.")
    else:
        reasoning_trace.append("No candidate fixes matched. Recommend manual review.")

    summary = (
        f"Shipment **{shipment_id}** | Carrier **{carrier}** | Vessel **{vessel}**\n"
        f"- Route: **{lane}**\n"
        f"- DQ Score: **{dq_score}** | Severity: **{severity}** | Confidence: **{confidence_label}** ({confidence_score:.2f})\n"
        f"- Top rules: **{', '.join(top_rules) if top_rules else 'None'}**"
    )

    ticket = {
        "type": "DQ_TRIAGE",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "shipment_id": shipment_id,
        "carrier": carrier,
        "vessel": vessel,
        "lane": lane,
        "dq_score": dq_score,
        "severity": severity,
        "confidence": confidence_label,
        "confidence_score": round(confidence_score, 2),
        "confidence_reasons": confidence_reasons,
        "triggered_rules": dq_details_sorted,
        "observations": observations,
        "likely_root_causes": likely_causes,
        "recommended_actions": actions,
        "suggested_owner": ", ".join(sorted(owners)) if owners else "Data Quality",
        "fix_simulation": fix_result,
        "reasoning_trace": reasoning_trace,
    }

    return {
        "summary_markdown": summary,
        "observations": observations,
        "reasoning_trace": reasoning_trace,
        "confidence_label": confidence_label,
        "confidence_score": confidence_score,
        "confidence_reasons": confidence_reasons,
        "likely_root_causes": likely_causes,
        "recommended_actions": actions,
        "suggested_owner": ticket["suggested_owner"],
        "fix_simulation": fix_result,
        "ticket": ticket,
    }


def run_batch_triage(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Run triage on worst shipments in a DataFrame.

    Args:
        df: DataFrame with processed_shipments_data.csv schema
        top_n: Number of worst shipments to triage

    Returns:
        DataFrame with triage results
    """
    # Score all shipments
    scores = []
    for idx, row in df.iterrows():
        scored = score_single_shipment(row.to_dict())
        scores.append({
            "idx": idx,
            "dq_score": scored["dq_score"],
            "max_severity": scored["max_severity"],
            "dq_flags": scored["dq_flags"],
        })

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.sort_values(
        ["dq_score", "max_severity"],
        ascending=[True, False]
    ).head(top_n)

    # Triage worst shipments
    results = []
    for idx in scores_df["idx"]:
        row = df.loc[idx].to_dict()
        triage = triage_shipment(row)
        ticket = triage["ticket"]
        fix = ticket.get("fix_simulation", {}) or {}

        results.append({
            "shipment_id": ticket["shipment_id"],
            "carrier": ticket["carrier"],
            "lane": ticket["lane"],
            "dq_score": ticket["dq_score"],
            "severity": ticket["severity"],
            "confidence": f"{ticket['confidence']} ({ticket['confidence_score']})",
            "top_rules": ", ".join([d.get("rule", "") for d in ticket.get("triggered_rules", [])[:3]]),
            "suggested_owner": ticket["suggested_owner"],
            "best_fix": fix.get("fix_name"),
            "fix_risk": fix.get("risk"),
            "score_after_fix": fix.get("score_after"),
            "improved": "Yes" if fix.get("improved") else "No",
        })

    return pd.DataFrame(results)


# --------------------------------------------------------
# CLI for testing
# --------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Default path
    data_path = "C:/Users/pelli/Downloads/processed_shipments_data.csv"

    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Parse dates
    for col in ["ETD", "ATD", "ETA", "ATA"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"Loaded {len(df)} shipments\n")

    # Score all shipments
    print("Scoring all shipments...")
    all_scores = []
    for idx, row in df.iterrows():
        scored = score_single_shipment(row.to_dict())
        all_scores.append(scored["dq_score"])

    df["dq_score"] = all_scores

    print(f"\nDQ Score Distribution:")
    print(f"  Mean: {df['dq_score'].mean():.1f}")
    print(f"  Min:  {df['dq_score'].min()}")
    print(f"  Max:  {df['dq_score'].max()}")
    print(f"  Critical (<60): {(df['dq_score'] < 60).sum()} ({(df['dq_score'] < 60).mean()*100:.1f}%)")
    print(f"  Clean (>90):    {(df['dq_score'] > 90).sum()} ({(df['dq_score'] > 90).mean()*100:.1f}%)")

    # Triage top 10 worst
    print("\n" + "="*60)
    print("TRIAGING TOP 10 WORST SHIPMENTS")
    print("="*60)

    results = run_batch_triage(df, top_n=10)
    print(results.to_string(index=False))

    # Show detailed triage for the worst one
    worst_idx = df["dq_score"].idxmin()
    worst_row = df.loc[worst_idx].to_dict()

    print("\n" + "="*60)
    print("DETAILED TRIAGE: WORST SHIPMENT")
    print("="*60)

    triage = triage_shipment(worst_row)
    print(triage["summary_markdown"])
    print("\nObservations:")
    for obs in triage["observations"]:
        print(f"  - {obs}")
    print("\nLikely Root Causes:")
    for cause in triage["likely_root_causes"]:
        print(f"  - {cause}")
    print("\nRecommended Actions:")
    for action in triage["recommended_actions"]:
        print(f"  - {action}")
    print(f"\nSuggested Owner: {triage['suggested_owner']}")

    fix = triage["fix_simulation"]
    if fix.get("attempted"):
        print(f"\nBest Fix: {fix['fix_name']} (Risk: {fix['risk']})")
        print(f"Score: {fix['score_before']} -> {fix['score_after']} ({'Improved' if fix['improved'] else 'No improvement'})")
