import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
from datetime import datetime, timezone
import uuid

from dq_engine import run_data_quality_checks
from agent_playbook import triage_shipment


# ---------------------------
# Helpers
# ---------------------------
def _dt(s):
    return pd.to_datetime(s, errors="coerce")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def month_season(m):
    if pd.isna(m):
        return "Unknown"
    m = int(m)
    if m in [11, 12]:
        return "Peak (Nov-Dec)"
    if m in [6, 7, 8]:
        return "Summer (Jun-Aug)"
    if m in [1, 2]:
        return "Post-peak (Jan-Feb)"
    return "Normal"

def compute_iqr_outliers(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return pd.Series([False] * len(series), index=series.index)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    return (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))

def normalize_congestion(x: pd.Series) -> pd.Series:
    """
    Your dataset route_congestion_index looks like 0..10 sometimes (e.g., 6.7).
    Normalize to 0..1 for the risk model.
    """
    s = pd.to_numeric(x, errors="coerce")
    if s.dropna().empty:
        return s
    if s.quantile(0.95) > 1.2:
        return (s / 10.0).clip(0, 1)
    return s.clip(0, 1)

@st.cache_data(show_spinner=False)
def normalize_schema(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    NEW schema → internal columns for dq_engine + agent:
      planned_pickup_ts, planned_delivery_ts, actual_delivery_ts, actual_pickup_ts, last_update_ts
    """
    df = df_raw.copy()

    # Parse dates
    df["departure_date"] = _dt(df["departure_date"])
    df["expected_arrival_date"] = _dt(df["expected_arrival_date"])
    df["actual_arrival_date"] = _dt(df["actual_arrival_date"])

    # Internal aliases
    df["origin"] = df["origin_port"]
    df["destination"] = df["destination_port"]
    df["lane"] = df["origin_port"].astype(str) + " → " + df["destination_port"].astype(str)

    df["planned_pickup_ts"] = df["departure_date"]
    df["planned_delivery_ts"] = df["expected_arrival_date"]
    df["actual_delivery_ts"] = df["actual_arrival_date"]

    # Ocean dataset usually lacks pickup milestones → treat pickup as departure
    df["actual_pickup_ts"] = df["departure_date"]

    # last_update_ts approximation from lag hours
    lag = pd.to_numeric(df["milestone_update_lag_hours"], errors="coerce")
    df["last_update_ts"] = df["actual_delivery_ts"] + pd.to_timedelta(lag.fillna(0), unit="h")

    # Derived delay
    df["delivery_delay_days"] = (df["actual_delivery_ts"] - df["planned_delivery_ts"]).dt.days
    df["delivery_delay_days"] = pd.to_numeric(df["delivery_delay_days"], errors="coerce").fillna(0)

    # Completeness
    filled = pd.to_numeric(df["mandatory_fields_filled_count"], errors="coerce").fillna(0)
    missing = pd.to_numeric(df["mandatory_fields_missing_count"], errors="coerce").fillna(0)
    total = (filled + missing).replace(0, np.nan)
    df["data_completeness_score"] = ((filled / total) * 100).fillna(0).round(1)

    # Planned vs delivered
    df["is_delivered"] = df["actual_delivery_ts"].notna()
    df["is_planned"] = ~df["is_delivered"]

    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    df["days_to_etd"] = (df["planned_pickup_ts"] - now).dt.days

    df["season"] = df["planned_pickup_ts"].dt.month.apply(month_season)
    df["week"] = df["planned_pickup_ts"].dt.to_period("W").astype(str)
    df["month"] = df["planned_pickup_ts"].dt.to_period("M").astype(str)

    return df

@st.cache_data(show_spinner=False)
def compute_ops_risk(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Explainable local risk (no external AI).
    """
    df = df_in.copy()

    cong = normalize_congestion(df["route_congestion_index"]).fillna(0)
    comp_gap = (100 - df["data_completeness_score"].fillna(100)).clip(0, 100) / 100.0
    carrier_hist = pd.to_numeric(df["carrier_historical_delay_rate"], errors="coerce").fillna(0).clip(0, 1)
    eta_rev = pd.to_numeric(df["eta_revision_count"], errors="coerce").fillna(0).clip(0, 20) / 20.0
    lag = pd.to_numeric(df["milestone_update_lag_hours"], errors="coerce").fillna(0).clip(0, 240) / 240.0
    dq_gap = (100 - df["dq_score"].fillna(100)).clip(0, 100) / 100.0

    z = (
        -1.1
        + 1.9 * comp_gap
        + 1.6 * cong
        + 1.2 * carrier_hist
        + 1.0 * eta_rev
        + 0.8 * lag
        + 1.4 * dq_gap
    )
    df["delay_risk_prob"] = sigmoid(z).round(4)

    bins = [0, 0.35, 0.60, 0.80, 1.0]
    labels = ["Low", "Medium", "High", "Critical"]
    df["urgency"] = pd.cut(df["delay_risk_prob"], bins=bins, labels=labels, include_lowest=True).astype(str)

    # Forecast transit = quote + uplift
    quote = pd.to_numeric(df["transit_time_days"], errors="coerce").fillna(7).clip(1, 90)
    uplift = (df["delay_risk_prob"] * 3.0).round(1)
    df["forecast_transit_days"] = (quote + uplift).round(1)
    df["forecast_arrival_date"] = df["planned_pickup_ts"] + pd.to_timedelta(df["forecast_transit_days"], unit="D")

    def explain_row(r):
        causes = []
        if r.get("data_completeness_score", 100) < 90:
            causes.append("Missing mandatory fields / low completeness")
        if normalize_congestion(pd.Series([r.get("route_congestion_index")])).iloc[0] > 0.70:
            causes.append("High route congestion")
        if r.get("carrier_historical_delay_rate", 0) > 0.35:
            causes.append("Carrier delay history is high")
        if r.get("eta_revision_count", 0) >= 3:
            causes.append("Multiple ETA revisions")
        if r.get("milestone_update_lag_hours", 0) > 24:
            causes.append("Late milestone updates (data freshness issue)")
        if r.get("dq_score", 100) < 80:
            causes.append("Low DQ score indicates unreliable inputs")
        return causes[:5] if causes else ["No strong driver (monitor)"]

    df["ml_root_causes"] = df.apply(explain_row, axis=1)
    return df


# ---------------------------
# Charts (horizontal labels)
# ---------------------------
def bar_chart(df, x, y, title):
    st.altair_chart(
        alt.Chart(df, title=title).mark_bar().encode(
            x=alt.X(x, axis=alt.Axis(labelAngle=0, title=x.split(":")[0])),
            y=alt.Y(y, axis=alt.Axis(title=y.split(":")[0])),
            tooltip=list(df.columns),
        ).properties(height=320),
        use_container_width=True
    )

def line_chart(df, x, y, title):
    st.altair_chart(
        alt.Chart(df, title=title).mark_line(point=True).encode(
            x=alt.X(x, axis=alt.Axis(labelAngle=0, title=x.split(":")[0])),
            y=alt.Y(y, axis=alt.Axis(title=y.split(":")[0])),
            tooltip=list(df.columns),
        ).properties(height=320),
        use_container_width=True
    )

def heatmap_chart(df, x, y, color, title):
    st.altair_chart(
        alt.Chart(df, title=title).mark_rect().encode(
            x=alt.X(x, axis=alt.Axis(labelAngle=0, title=x.split(":")[0])),
            y=alt.Y(y, axis=alt.Axis(labelAngle=0, title=y.split(":")[0])),
            color=alt.Color(color),
            tooltip=list(df.columns),
        ).properties(height=360),
        use_container_width=True
    )

def scatter_chart(df, x, y, color, title):
    st.altair_chart(
        alt.Chart(df, title=title).mark_circle(size=70, opacity=0.7).encode(
            x=alt.X(x, axis=alt.Axis(labelAngle=0, title=x.split(":")[0])),
            y=alt.Y(y, axis=alt.Axis(title=y.split(":")[0])),
            color=alt.Color(color),
            tooltip=list(df.columns),
        ).properties(height=320),
        use_container_width=True
    )


# ---------------------------
# Simulated actions
# ---------------------------
def render_action_card(action: dict):
    st.markdown(f"### ✅ {action.get('action_type')}")

    if "freshdesk" in action:
        fd = action["freshdesk"]
        st.write("**System:** Freshdesk (simulated)")
        st.write(f"**Subject:** {fd['subject']}")
        st.write(f"**Priority:** {fd['priority']}")
        st.write(f"**Assigned Group:** {fd['group']}")
        st.write("**What happens:** A support ticket will be created for the owning team.")
    elif "jira" in action:
        ji = action["jira"]
        st.write("**System:** Jira (simulated)")
        st.write(f"**Project:** {ji['project']}")
        st.write(f"**Issue Type:** {ji['issue_type']}")
        st.write(f"**Summary:** {ji['summary']}")
        st.write("**What happens:** A tracking issue will be logged for investigation.")
    elif "notification" in action:
        nt = action["notification"]
        st.write("**System:** Teams / Email (simulated)")
        st.write(f"**Recipient:** {nt['to']}")
        st.code(nt["message"])
    elif "autofix_proposal" in action:
        st.write("**System:** Auto-fix Proposal (simulated)")
        st.write("**Policy:** No automatic changes to production data.")
        st.write("**What happens:** A fix proposal is generated for review.")

    with st.expander("🔍 View technical payload (JSON)"):
        st.json(action)

def build_simulated_payload(action_type: str, ticket: dict) -> dict:
    now = datetime.utcnow().isoformat() + "Z"
    action_id = str(uuid.uuid4())

    base = {
        "action_id": action_id,
        "action_type": action_type,
        "created_at": now,
        "shipment_id": ticket.get("shipment_id"),
        "severity": ticket.get("severity"),
        "confidence": ticket.get("confidence"),
        "suggested_owner": ticket.get("suggested_owner"),
        "lane": ticket.get("lane"),
        "carrier": ticket.get("carrier"),
    }

    if action_type == "Freshdesk Ticket (simulated)":
        return {
            **base,
            "freshdesk": {
                "subject": f"[DQ] {ticket.get('severity')} | Shipment {ticket.get('shipment_id')} | Score {ticket.get('dq_score')}",
                "priority": "Urgent" if ticket.get("severity") == "HIGH" else "High",
                "group": ticket.get("suggested_owner", "Data Quality"),
                "description": {
                    "summary": "Auto-generated DQ triage ticket (simulated).",
                    "top_rules": [d.get("rule") for d in ticket.get("triggered_rules", [])[:5]],
                    "recommended_actions": ticket.get("recommended_actions", [])[:6],
                },
            },
        }

    if action_type == "Jira Issue (simulated)":
        return {
            **base,
            "jira": {
                "project": "DQOPS",
                "issue_type": "Bug" if ticket.get("severity") == "HIGH" else "Task",
                "summary": f"DQ issue: {ticket.get('shipment_id')} ({ticket.get('severity')})",
                "labels": ["data-quality", "ocean", f"sev-{str(ticket.get('severity','NONE')).lower()}"],
                "description": {
                    "dq_score": ticket.get("dq_score"),
                    "top_rules": [d.get("rule") for d in ticket.get("triggered_rules", [])[:5]],
                },
            },
        }

    if action_type == "Teams/Email Alert (simulated)":
        top_rules = ", ".join([d.get("rule") for d in ticket.get("triggered_rules", [])[:3]])
        return {
            **base,
            "notification": {
                "channel": "Teams",
                "to": ticket.get("suggested_owner", "Data Quality"),
                "message": (
                    f"🚨 DQ Alert ({ticket.get('severity')})\n"
                    f"Shipment: {ticket.get('shipment_id')} | Score: {ticket.get('dq_score')}\n"
                    f"Carrier: {ticket.get('carrier')} | Lane: {ticket.get('lane')}\n"
                    f"Top rules: {top_rules}"
                ),
            },
        }

    if action_type == "Auto-fix Proposal (simulated)":
        return {
            **base,
            "autofix_proposal": {
                "policy": "No automatic write-back in PoC. Proposals only.",
                "best_fix_summary": {
                    "fix_name": ticket.get("fix_simulation", {}).get("fix_name"),
                    "risk": ticket.get("fix_simulation", {}).get("risk"),
                    "score_before": ticket.get("fix_simulation", {}).get("score_before"),
                    "score_after": ticket.get("fix_simulation", {}).get("score_after"),
                },
                "requires_approval": True,
            },
        }

    return base


# ---------------------------
# Page
# ---------------------------
st.set_page_config(page_title="TMS Data Quality Analytics", layout="wide")
st.title("📦 TMS Data Quality Analytics Platform")

# st.markdown("""
# ## 🧱 Technology Stack
# - 🐍 **Python 3**
# - 📊 **Pandas & NumPy**
# - 🎛️ **Streamlit**
# - 📉 **Altair**
# - ✅ **YAML-based DQ rule engine**
# - 🧠 **Local risk scoring + explainable reasons**
# - 🤖 **Agentic triage + simulated orchestration actions**
# """)

st.markdown("""
### 🧱 Technology Stack & Intelligence Model

🐍 **Python 3**  
Core language — all logic, scoring, simulation runs locally.

📊 **Pandas & NumPy**  
Data processing, aggregations, delay calculations, risk scoring math.

🎛️ **Streamlit**  
Interactive dashboard UI — file upload, filters, tables, exports.

📉 **Altair**  
Clear visual analytics — heatmaps, trends, comparisons with readable axes.

✅ **YAML-based DQ rule engine**  
Explainable data quality rules with penalties and severities.

🧠 **Local risk scoring & explanations**  
Deterministic scoring using data completeness, congestion, carrier history, ETA churn.

🤖 **Agentic triage & simulated orchestration**  
A rule-driven agent that observes, scores, simulates fixes, and recommends actions  
(Jira / Freshdesk / Email are simulated — no real write-back).

---

### 🧠 Why this is agentic AI (without an external LLM)

This system uses a **deterministic, rule-driven agent**, not a probabilistic LLM.  
The intelligence comes from **agent orchestration**, not a neural network.

- Rules are defined in YAML  
- Decision logic is pure Python  
- Fixes are **simulated**, never auto-applied  

Everything runs **locally**, which is often preferred in enterprise data quality systems.

*If required, the same agent loop could later call an LLM — without changing the core logic.*
""")


st.divider()
uploaded_file = st.file_uploader("Upload shipment CSV (NEW schema)", type=["csv"])

if uploaded_file:
    with st.spinner("Loading & normalizing schema..."):
        df_raw = pd.read_csv(uploaded_file)
        df_norm = normalize_schema(df_raw)

    with st.spinner("Running DQ rules engine..."):
        dq_df = run_data_quality_checks(df_norm)

    with st.spinner("Computing operational risk signals..."):
        ops_df = compute_ops_risk(dq_df)

    # ==========================================================
    # OPS MANAGER STORY (CLEAN HEADINGS)
    # ==========================================================
    st.header("🧭 Operations Manager Dashboard")

    st.subheader("Data Quality & Impact Summary")
    tmp = ops_df.copy()

    hi = tmp[tmp["data_completeness_score"] >= 90]["delivery_delay_days"].mean()
    lo = tmp[tmp["data_completeness_score"] < 90]["delivery_delay_days"].mean()
    hi = 0 if pd.isna(hi) else float(hi)
    lo = 0 if pd.isna(lo) else float(lo)
    extra_per_ship = max(0.0, lo - hi)
    weak_count = int((tmp["data_completeness_score"] < 90).sum())
    est_extra_delay_days = round(extra_per_ship * weak_count, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg completeness", f"{ops_df['data_completeness_score'].mean():.1f}%")
    c2.metric("Avg DQ score", f"{ops_df['dq_score'].mean():.1f}")
    c3.metric("Weak data shipments (<90% complete)", weak_count)
    c4.metric("Est. delay-days caused by weak data", est_extra_delay_days)

    st.divider()

    st.subheader("ETD Countdown Risk Heatmap (7 to 1 days before ETD)")
    planned = ops_df[ops_df["is_planned"]].copy()

    # Window: 1..7 days before ETD (more intuitive and matches requirement)
    window = planned[(planned["days_to_etd"] >= 1) & (planned["days_to_etd"] <= 7)].copy()

    if window.empty:
        st.warning("No planned shipments in the 1–7 days ETD window. Use the demo dataset generator script.")
    else:
        window["etd_bucket"] = window["days_to_etd"].astype(int).astype(str)
        order = [str(x) for x in range(7, 0, -1)]  # 7..1
        grid = (
            window.groupby(["lane", "etd_bucket"])
            .agg(avg_risk=("delay_risk_prob", "mean"), shipments=("shipment_id", "count"))
            .reset_index()
        )
        grid["etd_bucket"] = pd.Categorical(grid["etd_bucket"], categories=order, ordered=True)
        heatmap_chart(grid, "etd_bucket:N", "lane:N", "avg_risk:Q", "Avg delay risk (lane × days to ETD)")

    st.divider()

    st.subheader("At-Risk Shipment List")
    risk_threshold = st.slider("Delay risk threshold", 0.10, 0.95, 0.45, 0.05)
    risk_list = planned[planned["delay_risk_prob"] >= risk_threshold].copy()
    risk_list = risk_list.sort_values(["delay_risk_prob", "days_to_etd"], ascending=[False, True])

    if risk_list.empty:
        st.warning("No planned shipments above threshold. Lower threshold or use demo dataset generator script.")
    else:
        view = risk_list.copy()
        view["delay_risk_prob"] = (view["delay_risk_prob"] * 100).round(1).astype(str) + "%"
        view["root_causes"] = view["ml_root_causes"].apply(lambda x: ", ".join(x[:3]) if isinstance(x, list) else str(x))
        cols = [
            "shipment_id", "origin_port", "destination_port", "lane", "carrier",
            "days_to_etd", "delay_risk_prob", "urgency",
            "data_completeness_score", "dq_score",
            "route_congestion_index", "carrier_historical_delay_rate",
            "transit_time_days", "forecast_transit_days", "forecast_arrival_date",
            "root_causes",
        ]
        st.dataframe(view[[c for c in cols if c in view.columns]].head(50), use_container_width=True)

    st.divider()

    st.subheader("Transit Time Forecast vs Quote")
    comp = planned[pd.notna(planned["transit_time_days"]) & pd.notna(planned["forecast_transit_days"])].copy()
    if comp.empty:
        st.warning("Not enough planned shipments to compare quote vs forecast. Use demo dataset generator script.")
    else:
        by = st.selectbox("Compare by", ["carrier", "lane", "origin_port"])
        agg = (
            comp.groupby(by)
            .agg(
                quoted=("transit_time_days", "mean"),
                forecast=("forecast_transit_days", "mean"),
                avg_risk=("delay_risk_prob", "mean"),
                shipments=("shipment_id", "count"),
            )
            .reset_index()
        )
        melt = agg.melt(id_vars=[by, "avg_risk", "shipments"], value_vars=["quoted", "forecast"], var_name="metric", value_name="days")
        chart = (
            alt.Chart(melt, title="Avg quoted vs forecast transit time")
            .mark_bar()
            .encode(
                x=alt.X(f"{by}:N", axis=alt.Axis(labelAngle=0, title=by), sort="-y"),
                y=alt.Y("days:Q", axis=alt.Axis(title="Days")),
                color="metric:N",
                tooltip=[by, "metric", "days", "avg_risk", "shipments"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    st.divider()

    st.subheader("ML Root Cause Explanation Panel")
    if planned.empty:
        st.warning("No planned shipments available. Use demo dataset generator script.")
    else:
        pick_id = st.selectbox("Select shipment", options=planned["shipment_id"].astype(str).tolist())
        sel = planned[planned["shipment_id"].astype(str) == str(pick_id)].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delay risk", f"{sel['delay_risk_prob']*100:.1f}%")
        c2.metric("Urgency", sel["urgency"])
        c3.metric("DQ score", f"{sel['dq_score']:.1f}")
        c4.metric("Completeness", f"{sel['data_completeness_score']:.1f}%")

        causes = sel["ml_root_causes"] if isinstance(sel["ml_root_causes"], list) else [str(sel["ml_root_causes"])]
        for c in causes[:6]:
            st.write(f"• {c}")

    st.divider()

    # ==========================================================
    # NEW REQUIREMENT: Bad shipments
    # ==========================================================
    st.header("📍 Bad Shipments Analysis")

    ops_df["is_bad_shipment"] = (
        (ops_df["dq_score"] < 80)
        | (ops_df["delay_risk_prob"] >= 0.60)
        | (normalize_congestion(ops_df["route_congestion_index"]).fillna(0) > 0.70)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Bad shipments (count)", int(ops_df["is_bad_shipment"].sum()))
    c2.metric("Bad shipments (%)", f"{(ops_df['is_bad_shipment'].mean()*100):.1f}%")
    c3.metric("Avg congestion (bad)", f"{normalize_congestion(ops_df.loc[ops_df['is_bad_shipment'], 'route_congestion_index']).mean():.2f}")

    st.subheader("Congestion per port (weekly / monthly)")
    freq = st.radio("View by", ["Weekly", "Monthly"], horizontal=True)
    key = "week" if freq == "Weekly" else "month"

    port_view = ops_df[ops_df["is_bad_shipment"]].copy()
    agg_port = (
        port_view.groupby([key, "origin_port"])
        .agg(
            bad_shipments=("shipment_id", "count"),
            avg_congestion=("route_congestion_index", "mean"),
            avg_delay_days=("delivery_delay_days", "mean"),
        )
        .reset_index()
        .sort_values(["bad_shipments"], ascending=False)
    )

    top_ports = agg_port.groupby("origin_port")["bad_shipments"].sum().sort_values(ascending=False).head(8).index.tolist()
    agg_port_top = agg_port[agg_port["origin_port"].isin(top_ports)]

    st.altair_chart(
        alt.Chart(agg_port_top, title=f"Avg congestion by origin port ({freq}) — bad shipments only")
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{key}:N", axis=alt.Axis(labelAngle=0, title=key)),
            y=alt.Y("avg_congestion:Q", axis=alt.Axis(title="Avg congestion index")),
            color="origin_port:N",
            tooltip=[key, "origin_port", "bad_shipments", "avg_congestion", "avg_delay_days"],
        )
        .properties(height=340),
        use_container_width=True
    )

    st.divider()

    st.subheader("Choice of carrier (season / route patterns)")

    metric = st.radio("Show metric", ["Bad shipment rate", "Bad shipment count"], horizontal=True)

    carrier_season = (
        ops_df.groupby(["season", "carrier"])
        .agg(
            shipments=("shipment_id", "count"),
            bad_shipments=("is_bad_shipment", "sum"),
            avg_risk=("delay_risk_prob", "mean"),
        )
        .reset_index()
    )
    carrier_season["bad_rate"] = (carrier_season["bad_shipments"] / carrier_season["shipments"]).fillna(0)

    top_carriers = carrier_season.groupby("carrier")["shipments"].sum().sort_values(ascending=False).head(10).index.tolist()
    cs_top = carrier_season[carrier_season["carrier"].isin(top_carriers)].copy()

    y_field = "bad_rate:Q" if metric == "Bad shipment rate" else "bad_shipments:Q"
    y_title = "Bad shipment rate" if metric == "Bad shipment rate" else "Bad shipments (count)"

    st.altair_chart(
        alt.Chart(cs_top, title="Carrier performance by season")
        .mark_bar()
        .encode(
            x=alt.X("carrier:N", axis=alt.Axis(labelAngle=0, title="carrier")),
            y=alt.Y(y_field, axis=alt.Axis(title=y_title)),
            color="season:N",
            tooltip=["carrier", "season", "shipments", "bad_shipments", "bad_rate", "avg_risk"],
        )
        .properties(height=340),
        use_container_width=True
    )

    route_carrier = (
        ops_df.groupby(["lane", "carrier"])
        .agg(
            shipments=("shipment_id", "count"),
            bad_shipments=("is_bad_shipment", "sum"),
            avg_risk=("delay_risk_prob", "mean"),
        )
        .reset_index()
    )
    route_carrier["bad_rate"] = (route_carrier["bad_shipments"] / route_carrier["shipments"]).fillna(0)

    top_lanes = route_carrier.groupby("lane")["bad_shipments"].sum().sort_values(ascending=False).head(8).index.tolist()
    rc_top = route_carrier[route_carrier["lane"].isin(top_lanes)].copy()

    st.altair_chart(
        alt.Chart(rc_top, title="Carrier performance for lanes with most bad shipments")
        .mark_bar()
        .encode(
            x=alt.X("lane:N", axis=alt.Axis(labelAngle=0, title="lane")),
            y=alt.Y(y_field, axis=alt.Axis(title=y_title)),
            color="carrier:N",
            tooltip=["lane", "carrier", "shipments", "bad_shipments", "bad_rate", "avg_risk"],
        )
        .properties(height=340),
        use_container_width=True
    )

    # ==========================================================
    # KEEP EVERYTHING (AGENTIC + TECHNICAL) AT THE END
    # ==========================================================
    st.header("🧩 Technical & Agentic Views (kept)")

    st.subheader("🔢 Key Data Quality KPIs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Shipments", len(dq_df))
    c2.metric("Avg DQ Score", round(dq_df["dq_score"].mean(), 1))
    c3.metric("Critical (<60)", f"{(dq_df['dq_score'] < 60).mean()*100:.1f}%")
    c4.metric("Clean (>90)", f"{(dq_df['dq_score'] > 90).mean()*100:.1f}%")
    c5.metric("Delay Outliers", int(dq_df["delay_outlier"].sum()) if "delay_outlier" in dq_df.columns else 0)

    st.divider()

    st.subheader("🤖 Agent Actions: Auto-triage & Recommendations")
    top_n = st.slider("How many worst shipments should the agent triage automatically?", 3, 30, 10)

    with st.spinner("Agent is triaging shipments..."):
        worst = dq_df.sort_values(["dq_score", "max_severity"], ascending=[True, False]).head(top_n).copy()
        triage_results = []
        triage_tickets = []
        triage_objects = {}

        for _, row in worst.iterrows():
            triage = triage_shipment(row.to_dict())
            ticket = triage["ticket"]
            triage_tickets.append(ticket)
            triage_objects[ticket["shipment_id"]] = triage

            fix = ticket.get("fix_simulation", {}) or {}
            triage_results.append({
                "shipment_id": ticket["shipment_id"],
                "dq_score": ticket["dq_score"],
                "severity": ticket["severity"],
                "confidence": f"{ticket['confidence']} ({ticket['confidence_score']})",
                "suggested_owner": ticket["suggested_owner"],
                "top_rules": ", ".join([d.get("rule","") for d in ticket.get("triggered_rules", [])[:3]]),
                "best_fix_found": "Yes" if fix.get("attempted") else "No",
                "score_after_best_fix": fix.get("score_after"),
                "improved?": "Yes" if fix.get("improved") else "No",
            })

    st.dataframe(pd.DataFrame(triage_results), use_container_width=True)

    st.download_button(
        "Download triage tickets (JSON)",
        data=json.dumps(triage_tickets, indent=2, default=str),
        file_name="dq_triage_tickets_topN.json",
        mime="application/json",
    )

    st.divider()

    st.subheader("🛠️ Simulated Actions Panel (Agent Orchestration)")
    if triage_tickets:
        chosen_id2 = st.selectbox(
            "Pick a shipment to run simulated actions on",
            options=[t["shipment_id"] for t in triage_tickets],
            key="sim_actions_pick",
        )
        chosen_ticket2 = next(t for t in triage_tickets if t["shipment_id"] == chosen_id2)

        actions = st.multiselect(
            "Select simulated actions",
            options=[
                "Freshdesk Ticket (simulated)",
                "Jira Issue (simulated)",
                "Teams/Email Alert (simulated)",
                "Auto-fix Proposal (simulated)",
            ],
            default=["Freshdesk Ticket (simulated)", "Teams/Email Alert (simulated)"],
        )

        if st.button("Run simulated actions"):
            with st.spinner("Generating action artifacts..."):
                artifacts = [build_simulated_payload(a, chosen_ticket2) for a in actions]
            st.success(f"Simulated {len(artifacts)} action(s).")
            for art in artifacts:
                render_action_card(art)

            st.download_button(
                "Download action artifacts (JSON)",
                data=json.dumps(artifacts, indent=2, default=str),
                file_name=f"agent_actions_{chosen_id2}.json",
                mime="application/json",
            )

    st.divider()

    st.subheader("🚦 Severity Overview")
    sev_df = dq_df["max_severity"].value_counts().reset_index()
    sev_df.columns = ["severity", "count"]
    bar_chart(sev_df, "severity:N", "count:Q", "Shipments by Severity")

    st.subheader("⏱️ Average Delivery Delay by Carrier (days)")
    carrier_delay = (
        dq_df.groupby("carrier")
        .agg(avg_delivery_delay_days=("delivery_delay_days", "mean"))
        .reset_index()
        .sort_values("avg_delivery_delay_days")
    )
    bar_chart(carrier_delay, "carrier:N", "avg_delivery_delay_days:Q", "Average Delivery Delay by Carrier")

    st.subheader("📈 Average Delivery Delay Over Time")
    dq_df["pickup_date"] = pd.to_datetime(dq_df["planned_pickup_ts"], errors="coerce").dt.date
    delay_trend = (
        dq_df.groupby("pickup_date")
        .agg(avg_delivery_delay_days=("delivery_delay_days", "mean"))
        .reset_index()
        .dropna()
    )
    line_chart(delay_trend, "pickup_date:T", "avg_delivery_delay_days:Q", "Average Delivery Delay Over Time")

    st.subheader("🚨 Outlier Detection — Delay vs DQ Score")
    scatter_df = dq_df[pd.notna(dq_df["delivery_delay_days"])]
    if len(scatter_df) > 0:
        scatter_chart(
            scatter_df.sample(min(len(scatter_df), 1000), random_state=42),
            "delivery_delay_days:Q",
            "dq_score:Q",
            "delay_outlier:N",
            "Delivery Delay vs DQ Score (Outliers Highlighted)",
        )

    st.divider()

    st.subheader("📤 Export Problematic Shipments")
    export_threshold = st.slider("Export shipments with DQ Score below:", 0, 100, 70)
    export_df = dq_df[(dq_df["dq_score"] < export_threshold) | (dq_df["delay_outlier"])].copy()
    st.write(f"Shipments to export: {len(export_df)}")
    st.download_button(
        "Download flagged shipments (CSV)",
        data=export_df.to_csv(index=False),
        file_name="flagged_shipments.csv",
        mime="text/csv",
    )

else:
    st.info("Upload your CSV to generate Ops dashboard + agentic triage.")
