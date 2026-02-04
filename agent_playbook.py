from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import datetime
import copy

from dq_engine import score_single_shipment

PLAYBOOK: Dict[str, Dict[str, Any]] = {
    "MISSING_ACTUAL_DELIVERY": {
        "meaning": "Actual delivery timestamp is missing; delivery KPIs cannot be computed reliably.",
        "owner": "Integration / Data Engineering",
        "fixes": [
            "Backfill actual_delivery_ts from milestone feed if available",
            "Confirm shipment status is DELIVERED; if yes, re-ingest milestone events",
            "Check upstream mapping for actual_delivery_ts"
        ],
        "causes": [
            "Carrier/EDI milestone not received",
            "Shipment not finalized in TMS",
            "Field mapping issue"
        ],
    },
    "DELIVERY_BEFORE_PICKUP": {
        "meaning": "Delivery earlier than pickup is logically impossible.",
        "owner": "Data Engineering / Carrier Integration",
        "fixes": [
            "Check timezone normalization (UTC vs local)",
            "Validate event ordering in raw milestone feed",
            "If timestamps appear swapped, swap actual_pickup_ts and actual_delivery_ts"
        ],
        "causes": [
            "Timezone conversion issue",
            "Swapped timestamps",
            "Incorrect event sequencing from source"
        ],
    },
    "INVALID_TRANSIT_TIME": {
        "meaning": "Transit time outside policy bounds (<0 or >30 days).",
        "owner": "Data Quality / Ops Analytics",
        "fixes": [
            "Inspect pickup/delivery timestamps and milestone pairing",
            "Apply sanity bounds and route to manual review if outside policy"
        ],
        "causes": [
            "Bad timestamps",
            "Wrong milestone pairing",
            "Data entry error"
        ],
    },
    "LATE_UPDATE": {
        "meaning": "Update arrived >24 hours after delivery; timeliness SLA breach.",
        "owner": "Operations / Integration",
        "fixes": [
            "Review integration schedule / batch delays",
            "Add SLA monitoring for update_delay_hours"
        ],
        "causes": [
            "Late carrier status push",
            "Batch delay in integration",
            "TMS processing lag"
        ],
    },
    "MISSING_CARRIER": {
        "meaning": "Carrier missing; reduces ability to analyze carrier performance and routing.",
        "owner": "Operations / Master Data",
        "fixes": [
            "Enforce carrier as required at booking",
            "Backfill from booking reference / routing plan",
        ],
        "causes": [
            "Manual entry skipped field",
            "Carrier not assigned at planning stage",
            "Upstream mapping missing"
        ],
    },
}

SEV_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}

def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _observations_from_row(r: dict) -> List[str]:
    obs = []
    if r.get("dq_flags"):
        obs.append(f"Triggered rules: {r.get('dq_flags')}")
    if bool(r.get("delay_outlier")):
        obs.append("delivery_delay_days flagged as outlier (IQR)")
    if r.get("delivery_delay_days") is not None:
        obs.append(f"delivery_delay_days = {r.get('delivery_delay_days')}")
    if r.get("pickup_delay_days") is not None:
        obs.append(f"pickup_delay_days = {r.get('pickup_delay_days')}")
    if r.get("update_delay_hours") is not None:
        obs.append(f"update_delay_hours = {r.get('update_delay_hours')}")
    return obs

def _confidence_from_evidence(dq_details: List[dict], severity: str) -> Tuple[float, List[str]]:
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
# ✅ MULTI-CANDIDATE FIX GENERATOR (Upgrade C)
# --------------------------------------------------------
def _build_fix_candidates(row: dict, before: dict) -> List[dict]:
    """
    Return list of candidates:
      { name, apply_fn, risk, notes }
    where apply_fn(record)->(new_record, proposed_changes)
    """
    triggered = [d.get("rule") for d in before.get("dq_details", [])]
    candidates = []

    # Helper to format changes
    def change(from_val, to_val):
        return {"from": from_val, "to": to_val}

    # Candidate 1: Swap pickup/delivery if DELIVERY_BEFORE_PICKUP
    if "DELIVERY_BEFORE_PICKUP" in triggered:
        def apply_swap(rec):
            rec2 = copy.deepcopy(rec)
            ap = rec2.get("actual_pickup_ts")
            ad = rec2.get("actual_delivery_ts")
            if ap is None or ad is None:
                return rec2, {}
            rec2["actual_pickup_ts"], rec2["actual_delivery_ts"] = ad, ap
            return rec2, {
                "actual_pickup_ts": change(ap, rec2["actual_pickup_ts"]),
                "actual_delivery_ts": change(ad, rec2["actual_delivery_ts"]),
            }
        candidates.append({
            "name": "Swap actual_pickup_ts and actual_delivery_ts",
            "risk": "MEDIUM",
            "notes": "Common cause: swapped events or timezone mis-handling. Validate against raw milestones.",
            "apply_fn": apply_swap
        })

        # Candidate 2: Add a timezone offset to actual_delivery_ts (simulate normalization)
        # We try small set of plausible offsets used in logistics regions
        # This is a *proposal* only; never auto-writeback.
        offsets_hours = [-8, -5, -4, 4, 5, 8]
        def apply_offset_factory(h):
            def apply_offset(rec):
                rec2 = copy.deepcopy(rec)
                ad = rec2.get("actual_delivery_ts")
                if ad is None:
                    return rec2, {}
                try:
                    rec2["actual_delivery_ts"] = ad + pd.Timedelta(hours=h)  # type: ignore
                except Exception:
                    return rec2, {}
                return rec2, {
                    "actual_delivery_ts": change(ad, rec2["actual_delivery_ts"]),
                    "_note": f"Shifted actual_delivery_ts by {h} hours (timezone normalization attempt)"
                }
            return apply_offset

        # Import pandas only if needed (to avoid dependency issues)
        import pandas as pd  # local import
        for h in offsets_hours:
            candidates.append({
                "name": f"Timezone normalize: shift actual_delivery_ts by {h}h",
                "risk": "HIGH",
                "notes": "Only apply if your pipeline confirms a consistent timezone offset problem.",
                "apply_fn": apply_offset_factory(h)
            })

    # Candidate 3: If carrier missing, set carrier=UNKNOWN (safe)
    if "MISSING_CARRIER" in triggered:
        def apply_carrier_unknown(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("carrier")
            rec2["carrier"] = "UNKNOWN"
            return rec2, {"carrier": change(old, "UNKNOWN")}
        candidates.append({
            "name": "Set missing carrier to 'UNKNOWN' (placeholder)",
            "risk": "LOW",
            "notes": "Safe placeholder to remove nulls; should be replaced when master data is available.",
            "apply_fn": apply_carrier_unknown
        })

    # Candidate 4: If actual delivery missing, propose backfill from planned_delivery_ts
    # This can improve DQ score but is potentially misleading; mark as HIGH risk.
    if "MISSING_ACTUAL_DELIVERY" in triggered:
        def apply_backfill_delivery_from_planned(rec):
            rec2 = copy.deepcopy(rec)
            old = rec2.get("actual_delivery_ts")
            pdv = rec2.get("planned_delivery_ts")
            if pdv is None:
                return rec2, {}
            rec2["actual_delivery_ts"] = pdv
            return rec2, {"actual_delivery_ts": change(old, pdv), "_note": "Backfilled actual_delivery_ts from planned_delivery_ts"}
        candidates.append({
            "name": "Backfill actual_delivery_ts from planned_delivery_ts (proposal)",
            "risk": "HIGH",
            "notes": "Only valid if shipment is confirmed delivered and planned≈actual is acceptable for your KPI policy.",
            "apply_fn": apply_backfill_delivery_from_planned
        })

    # Candidate 5: If transit invalid, align actual_delivery_ts to actual_pickup_ts + planned_transit_days
    # This is a best-effort estimate; still a proposal only.
    if "INVALID_TRANSIT_TIME" in triggered:
        def apply_align_delivery_using_planned_transit(rec):
            rec2 = copy.deepcopy(rec)
            ap = rec2.get("actual_pickup_ts")
            pp = rec2.get("planned_pickup_ts")
            pdv = rec2.get("planned_delivery_ts")
            old = rec2.get("actual_delivery_ts")

            # Prefer planned transit days if available, otherwise use planned_delivery_ts
            if ap is not None and pp is not None and pdv is not None:
                planned_days = (pdv - pp).days
                if planned_days is not None:
                    rec2["actual_delivery_ts"] = ap + (pdv - pp)
                    return rec2, {
                        "actual_delivery_ts": change(old, rec2["actual_delivery_ts"]),
                        "_note": "Aligned actual_delivery_ts using planned transit duration"
                    }

            if pdv is not None:
                rec2["actual_delivery_ts"] = pdv
                return rec2, {"actual_delivery_ts": change(old, pdv), "_note": "Fallback align: actual_delivery_ts := planned_delivery_ts"}

            return rec2, {}
        candidates.append({
            "name": "Align actual_delivery_ts using planned transit duration (proposal)",
            "risk": "HIGH",
            "notes": "Use only for recovery pipelines; keep audit tag that timestamp was estimated.",
            "apply_fn": apply_align_delivery_using_planned_transit
        })

    return candidates

def _simulate_fix_and_rescore(row: dict, rules_path="dq_rules.yaml") -> dict:
    """
    Agent loop:
      - Score baseline (tool)
      - Generate fix candidates (plan)
      - Apply each candidate (act)
      - Re-score each candidate (evaluate)
      - Pick best improvement (choose)
    """
    before = score_single_shipment(row, rules_path=rules_path)
    base_score = before["dq_score"]

    results = []
    candidates = _build_fix_candidates(row, before)

    for c in candidates:
        new_rec, proposed_changes = c["apply_fn"](row)
        after = score_single_shipment(new_rec, rules_path=rules_path)

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
        "candidates_evaluated": results_sorted[:12],  # keep top 12 candidates for transparency
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

def triage_shipment(row: dict, rules_path="dq_rules.yaml") -> dict:
    shipment_id = row.get("shipment_id", "UNKNOWN")
    carrier = row.get("carrier", "NA")
    lane = f"{row.get('origin','NA')} → {row.get('destination','NA')}"

    dq_details: List[dict] = row.get("dq_details", []) or []
    severity = row.get("max_severity", "NONE")
    dq_score = row.get("dq_score", None)

    dq_details_sorted = sorted(
        dq_details,
        key=lambda d: (SEV_RANK.get(d.get("severity","NONE"), 0), d.get("penalty", 0)),
        reverse=True
    )

    top_rules = [d.get("rule") for d in dq_details_sorted if d.get("rule")]
    top_rules = top_rules[:5]

    observations = _observations_from_row(row)

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

    fix_result = _simulate_fix_and_rescore(row, rules_path=rules_path)

    reasoning_trace = [
        f"Loaded evidence for shipment_id={shipment_id}, carrier={carrier}, lane={lane}",
        f"Observed dq_score={dq_score}, severity={severity}",
        f"Rules triggered (ordered) = {', '.join(top_rules) if top_rules else 'None'}",
        f"Confidence computed from severity + number of rules: {confidence_score:.2f} ({confidence_label})",
        f"Generated {len(fix_result.get('candidates_evaluated', []))} evaluated fix candidate(s) (showing top ones).",
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
        f"Shipment **{shipment_id}** | Carrier **{carrier}** | Lane **{lane}**\n"
        f"- DQ Score: **{dq_score}** | Severity: **{severity}** | Confidence: **{confidence_label}** ({confidence_score:.2f})\n"
        f"- Top rules: **{', '.join(top_rules) if top_rules else 'None'}**"
    )

    ticket = {
        "type": "DQ_TRIAGE",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "shipment_id": shipment_id,
        "carrier": carrier,
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
