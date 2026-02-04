"""
TMS Data Quality Analytics Platform V2
Adapted for processed_shipments_data.csv schema
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import requests
from datetime import datetime, timezone
import uuid

# ---------------------------
# Webhook Configuration
# ---------------------------
WEBHOOK_URL = "https://workflows.platform.happyrobot.ai/hooks/4rvbd0lqq4w6"

from agent_playbook_v2 import (
    triage_shipment,
    score_single_shipment,
    run_batch_triage,
    DQ_RULES,
    PLAYBOOK,
)

# ---------------------------
# Color Schemes
# ---------------------------
SEVERITY_COLORS = {
    "HIGH": "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW": "#3498db",
    "NONE": "#2ecc71"
}

SCORE_COLORS = {
    "Critical (0-40)": "#e74c3c",
    "Poor (40-60)": "#e67e22",
    "Fair (60-80)": "#f1c40f",
    "Good (80-90)": "#27ae60",
    "Excellent (90-100)": "#2ecc71"
}


# ---------------------------
# Webhook Functions
# ---------------------------
def build_carrier_alert_payload(carrier_name: str, shipments: list) -> dict:
    """Build a payload for a single carrier with all their critical shipments."""

    # Build list of shipment details
    shipment_records = []
    all_missing_fields = set()

    for ship in shipments:
        missing = []
        if pd.isna(ship.get("ATA")):
            missing.append("Actual Arrival Date (ATA)")
        if pd.isna(ship.get("ATD")):
            missing.append("Actual Departure Date (ATD)")
        if pd.isna(ship.get("ETA")):
            missing.append("Estimated Arrival Date (ETA)")
        if pd.isna(ship.get("Vessel")) or str(ship.get("Vessel", "")).strip() == "":
            missing.append("Vessel Name")

        all_missing_fields.update(missing)

        shipment_records.append({
            "shipment_id": ship.get("ID"),
            "origin": ship.get("POL"),
            "destination": ship.get("POD"),
            "vessel": ship.get("Vessel") if pd.notna(ship.get("Vessel")) else "MISSING",
            "etd": str(ship.get("ETD"))[:10] if pd.notna(ship.get("ETD")) else "MISSING",
            "eta": str(ship.get("ETA"))[:10] if pd.notna(ship.get("ETA")) else "MISSING",
            "atd": str(ship.get("ATD"))[:10] if pd.notna(ship.get("ATD")) else "MISSING",
            "ata": str(ship.get("ATA"))[:10] if pd.notna(ship.get("ATA")) else "MISSING",
            "dq_score": ship.get("dq_score"),
            "issues": ship.get("dq_flags"),
            "missing_fields": missing,
        })

    payload = {
        "event_type": "CARRIER_DATA_QUALITY_ALERT",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

        # Carrier information
        "carrier": {
            "name": carrier_name,
            "total_critical_shipments": len(shipments),
            "common_missing_fields": list(all_missing_fields),
        },

        # All shipments for this carrier
        "shipments": shipment_records,

        # Instructions for HappyRobot workflow
        "workflow_instructions": {
            "step_1_email": {
                "action": "SEND_EMAIL",
                "recipient": f"{carrier_name} Operations Team",
                "subject": f"URGENT: Data Quality Issues - {len(shipments)} Shipments Require Immediate Attention",
                "email_content_instructions": f"""
Send a professional email to {carrier_name} with the following structure:

1. GREETING: Address the carrier's operations team professionally

2. URGENCY: Explain that we have identified {len(shipments)} shipments with critical data quality issues that require immediate attention

3. ISSUE SUMMARY:
   - Total affected shipments: {len(shipments)}
   - Common missing data: {', '.join(all_missing_fields) if all_missing_fields else 'timestamp discrepancies'}

4. SHIPMENT LIST: Include a table or list of all affected shipments with:
   - Shipment ID
   - Route (Origin -> Destination)
   - What data is missing for each

5. ACTION REQUIRED: Request the carrier to:
   - Review each shipment listed
   - Provide the missing information (actual arrival dates, departure dates, vessel names)
   - Reply to this email with the corrected data within 24 hours

6. CONTACT: Provide contact information for follow-up questions

7. CLOSING: Thank them for their prompt attention to this matter
""",
            },
            "step_2_call": {
                "action": "MAKE_PHONE_CALL",
                "timing": "Immediately after sending the email",
                "call_script_instructions": f"""
Make a follow-up phone call to {carrier_name} with this script:

1. INTRODUCTION:
   "Hello, this is [Agent Name] calling from the TMS Operations team regarding urgent data quality issues."

2. PURPOSE:
   "I'm calling to follow up on an email we just sent regarding {len(shipments)} shipments that have critical missing information in our system."

3. URGENCY:
   "We need your team to review this email as soon as possible. The missing data is affecting our ability to track deliveries and provide accurate reporting."

4. SPECIFIC ASK:
   "Could you please confirm that someone from your operations team will review the email and provide the missing shipment data within the next 24 hours?"

5. MISSING DATA SUMMARY:
   "The main issues are: {', '.join(all_missing_fields) if all_missing_fields else 'incorrect or missing timestamps'}"

6. NEXT STEPS:
   "Once you receive the email, please reply with the corrected information. If you have any questions, you can reach us at this number."

7. CONFIRMATION:
   "Can I get your name and confirm you'll pass this along to the right team?"

8. CLOSING:
   "Thank you for your prompt attention. Have a great day."
""",
            },
        },

        # Summary message
        "summary": f"Critical data quality alert for {carrier_name}: {len(shipments)} shipments have missing or incorrect data. Workflow should send email with full details, then immediately call to ensure urgent follow-up.",
    }

    return payload


def send_carrier_alerts_webhook(carriers_data: dict) -> list:
    """Send webhooks for each carrier with their critical shipments."""
    results = []

    for carrier_name, shipments in carriers_data.items():
        payload = build_carrier_alert_payload(carrier_name, shipments)

        try:
            response = requests.post(
                WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            results.append({
                "carrier": carrier_name,
                "shipment_count": len(shipments),
                "success": response.status_code in [200, 201, 202],
                "status_code": response.status_code,
                "response": response.text[:200] if response.text else "",
                "payload": payload
            })
        except Exception as e:
            results.append({
                "carrier": carrier_name,
                "shipment_count": len(shipments),
                "success": False,
                "error": str(e),
                "payload": payload
            })

    return results


# ---------------------------
# Helpers
# ---------------------------
def _dt(s):
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner=False)
def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load CSV and normalize schema."""
    df = pd.read_csv(uploaded_file)

    # Parse dates
    for col in ["ETD", "ATD", "ETA", "ATA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numeric columns
    df["transit_days"] = pd.to_numeric(df.get("transit_days"), errors="coerce")
    df["ContainerCount"] = pd.to_numeric(df.get("ContainerCount"), errors="coerce")
    df["TotalWeight"] = pd.to_numeric(df.get("TotalWeight"), errors="coerce")
    df["TotalVolume"] = pd.to_numeric(df.get("TotalVolume"), errors="coerce")

    # Create lane
    df["lane"] = df["POL"].astype(str) + " -> " + df["POD"].astype(str)

    # Calculate delay_days
    df["delay_days"] = (df["ATA"] - df["ETA"]).dt.total_seconds() / 86400

    # Is delivered?
    df["is_delivered"] = df["ATA"].notna()

    return df


@st.cache_data(show_spinner=False)
def run_dq_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Score all shipments."""
    df = df.copy()

    scores = []
    severities = []
    flags_list = []
    details_list = []

    for idx, row in df.iterrows():
        scored = score_single_shipment(row.to_dict())
        scores.append(scored["dq_score"])
        severities.append(scored["max_severity"])
        flags_list.append(scored["dq_flags"])
        details_list.append(scored["dq_details"])

    df["dq_score"] = scores
    df["max_severity"] = severities
    df["dq_flags"] = flags_list
    df["dq_details"] = details_list

    # Score category
    def score_category(s):
        if s <= 40:
            return "Critical (0-40)"
        elif s <= 60:
            return "Poor (40-60)"
        elif s <= 80:
            return "Fair (60-80)"
        elif s <= 90:
            return "Good (80-90)"
        else:
            return "Excellent (90-100)"

    df["score_category"] = df["dq_score"].apply(score_category)

    # IQR outliers on delay_days
    s = pd.to_numeric(df["delay_days"], errors="coerce")
    if not s.dropna().empty:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        df["delay_outlier"] = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))
    else:
        df["delay_outlier"] = False

    return df


# ---------------------------
# Charts
# ---------------------------
def pie_chart(df, theta, color, title, color_scale=None):
    """Create a donut/pie chart."""
    base = alt.Chart(df, title=title).mark_arc(innerRadius=50, outerRadius=120).encode(
        theta=alt.Theta(theta, stack=True),
        color=alt.Color(
            color,
            scale=alt.Scale(domain=list(color_scale.keys()), range=list(color_scale.values())) if color_scale else alt.Undefined,
            legend=alt.Legend(title=None)
        ),
        tooltip=[color, theta]
    ).properties(height=300)

    st.altair_chart(base, use_container_width=True)


def horizontal_bar_chart(df, x, y, title, color=None, color_scale=None):
    """Horizontal bar chart - better for long labels."""
    encoding = {
        "x": alt.X(x, axis=alt.Axis(title=x.split(":")[0])),
        "y": alt.Y(y, axis=alt.Axis(title=None), sort="-x"),
        "tooltip": list(df.columns),
    }

    if color:
        if color_scale:
            encoding["color"] = alt.Color(
                color,
                scale=alt.Scale(domain=list(color_scale.keys()), range=list(color_scale.values())),
                legend=None
            )
        else:
            encoding["color"] = alt.Color(color, legend=None)

    chart = alt.Chart(df, title=title).mark_bar().encode(**encoding).properties(height=max(200, len(df) * 25))
    st.altair_chart(chart, use_container_width=True)


def bar_chart(df, x, y, title, color=None):
    encoding = {
        "x": alt.X(x, axis=alt.Axis(labelAngle=-45, title=x.split(":")[0])),
        "y": alt.Y(y, axis=alt.Axis(title=y.split(":")[0])),
        "tooltip": list(df.columns),
    }
    if color:
        encoding["color"] = alt.Color(color, legend=None)

    st.altair_chart(
        alt.Chart(df, title=title).mark_bar().encode(**encoding).properties(height=320),
        use_container_width=True
    )


def scatter_chart(df, x, y, color, title, color_scale=None):
    encoding = {
        "x": alt.X(x, axis=alt.Axis(title=x.split(":")[0])),
        "y": alt.Y(y, axis=alt.Axis(title=y.split(":")[0])),
        "tooltip": list(df.columns),
    }

    if color_scale:
        encoding["color"] = alt.Color(
            color,
            scale=alt.Scale(domain=list(color_scale.keys()), range=list(color_scale.values()))
        )
    else:
        encoding["color"] = alt.Color(color)

    st.altair_chart(
        alt.Chart(df, title=title).mark_circle(size=70, opacity=0.7).encode(**encoding).properties(height=350),
        use_container_width=True
    )


# ---------------------------
# Simulated Actions
# ---------------------------
def render_action_card(action: dict):
    st.markdown(f"### {action.get('action_type')}")

    if "freshdesk" in action:
        fd = action["freshdesk"]
        st.write("**System:** Freshdesk (simulated)")
        st.write(f"**Subject:** {fd['subject']}")
        st.write(f"**Priority:** {fd['priority']}")
        st.write(f"**Assigned Group:** {fd['group']}")
    elif "jira" in action:
        ji = action["jira"]
        st.write("**System:** Jira (simulated)")
        st.write(f"**Project:** {ji['project']}")
        st.write(f"**Issue Type:** {ji['issue_type']}")
        st.write(f"**Summary:** {ji['summary']}")
    elif "notification" in action:
        nt = action["notification"]
        st.write("**System:** Teams / Email (simulated)")
        st.write(f"**Recipient:** {nt['to']}")
        st.code(nt["message"])

    with st.expander("View technical payload (JSON)"):
        st.json(action)


def build_simulated_payload(action_type: str, ticket: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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
                "labels": ["data-quality", "shipping", f"sev-{str(ticket.get('severity','NONE')).lower()}"],
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
                    f"DQ Alert ({ticket.get('severity')})\n"
                    f"Shipment: {ticket.get('shipment_id')} | Score: {ticket.get('dq_score')}\n"
                    f"Carrier: {ticket.get('carrier')} | Lane: {ticket.get('lane')}\n"
                    f"Top rules: {top_rules}"
                ),
            },
        }

    return base


# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="TMS Data Quality Analytics V2", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
    }
    .critical-badge {
        background-color: #e74c3c;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .good-badge {
        background-color: #2ecc71;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("TMS Data Quality Analytics Platform")

# Sidebar for filters and info
with st.sidebar:
    st.header("Settings")

    use_default = st.checkbox("Use default data file", value=True)

    st.divider()
    st.header("About")
    st.markdown("""
    **Agent-Powered DQ Platform**

    - Rule-based scoring
    - Automated fix simulation
    - Triage recommendations
    - Export capabilities
    """)

    st.divider()
    st.markdown("**Tech Stack:** Python, Pandas, Streamlit, Altair")

# Load data
if use_default:
    data_path = "C:/Users/pelli/Downloads/processed_shipments_data.csv"
    try:
        with st.spinner("Loading data..."):
            df = load_and_process_data(data_path)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload shipment CSV", type=["csv"])
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_and_process_data(uploaded_file)
    else:
        st.info("Upload a CSV file to begin analysis.")
        st.stop()

# Run DQ scoring
with st.spinner("Running DQ scoring engine..."):
    df = run_dq_scoring(df)

# ==========================================================
# QUICK STATS BANNER
# ==========================================================
st.success(f"Loaded and scored **{len(df):,}** shipments")

# Key metrics in colored cards
col1, col2, col3, col4, col5 = st.columns(5)

avg_score = df['dq_score'].mean()
critical_count = (df['dq_score'] < 60).sum()
critical_pct = (df['dq_score'] < 60).mean() * 100
clean_count = (df['dq_score'] > 90).sum()
clean_pct = (df['dq_score'] > 90).mean() * 100

col1.metric("Total Shipments", f"{len(df):,}")
col2.metric("Avg DQ Score", f"{avg_score:.1f}", delta=f"{'Good' if avg_score > 80 else 'Needs Work'}")
col3.metric("Critical Issues", f"{critical_count:,}", delta=f"{critical_pct:.1f}%", delta_color="inverse")
col4.metric("Clean Records", f"{clean_count:,}", delta=f"{clean_pct:.1f}%")
col5.metric("Delay Outliers", f"{int(df['delay_outlier'].sum()):,}")

st.divider()

# ==========================================================
# CRITICAL ISSUES - ESCALATION CENTER
# ==========================================================
if critical_count > 0:
    st.header("Critical Issues - Carrier Escalation Center")

    st.warning(f"**{critical_count:,}** shipments have critical data quality issues (DQ Score < 60)")

    # Get critical shipments
    critical_df = df[df["dq_score"] < 60].copy()
    critical_df = critical_df.sort_values("dq_score", ascending=True)

    # Group by carrier
    # Handle missing carriers
    critical_df["Carrier_Display"] = critical_df["Carrier"].fillna("UNKNOWN CARRIER")

    carrier_summary = critical_df.groupby("Carrier_Display").agg(
        shipment_count=("ID", "count"),
        avg_score=("dq_score", "mean"),
        shipment_ids=("ID", list)
    ).reset_index().sort_values("shipment_count", ascending=False)

    # Display carrier summary
    st.subheader("Critical Issues by Carrier")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            carrier_summary[["Carrier_Display", "shipment_count", "avg_score"]].rename(columns={
                "Carrier_Display": "Carrier",
                "shipment_count": "Critical Shipments",
                "avg_score": "Avg DQ Score"
            }).head(15),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("**Escalation Workflow:**")
        st.markdown("""
        1. Select carrier(s) to escalate
        2. Click "Send Alerts"
        3. HappyRobot will:
           - **Email** carrier with list of issues
           - **Call** carrier to request immediate action
        """)

    st.markdown("---")

    # Escalation options
    escalation_mode = st.radio(
        "Escalation Mode:",
        ["Single Carrier", "All Carriers with Critical Issues"],
        horizontal=True
    )

    if escalation_mode == "Single Carrier":
        col1, col2 = st.columns([2, 1])

        with col1:
            # Get carriers with issues (exclude UNKNOWN for single selection)
            carrier_options = carrier_summary["Carrier_Display"].tolist()
            selected_carrier = st.selectbox(
                "Select carrier to escalate:",
                options=carrier_options
            )

            # Show shipments for selected carrier
            carrier_shipments = critical_df[critical_df["Carrier_Display"] == selected_carrier]
            st.markdown(f"**{len(carrier_shipments)} shipments will be included in the alert:**")

            display_cols = ["ID", "POL", "POD", "dq_score", "dq_flags"]
            st.dataframe(
                carrier_shipments[display_cols].head(10).rename(columns={
                    "ID": "Shipment ID",
                    "dq_score": "DQ Score",
                    "dq_flags": "Issues"
                }),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("")
            st.markdown("")
            st.markdown("")

            if st.button("Send Email + Call Alert", type="primary", use_container_width=True):
                with st.spinner(f"Sending alert for {selected_carrier} to HappyRobot..."):
                    # Prepare carrier data
                    carrier_data = {
                        selected_carrier: carrier_shipments.to_dict('records')
                    }
                    results = send_carrier_alerts_webhook(carrier_data)

                result = results[0] if results else {"success": False, "error": "No result"}

                if result["success"]:
                    st.success(f"Alert sent successfully for {selected_carrier}!")
                    st.info(f"HappyRobot will email {selected_carrier} with {len(carrier_shipments)} shipments, then make a follow-up call.")

                    with st.expander("View webhook payload sent to HappyRobot"):
                        st.json(result["payload"])
                else:
                    st.error(f"Failed to send alert: {result.get('error', 'Unknown error')}")
                    with st.expander("View details"):
                        st.json(result)

    else:  # All Carriers
        # Count carriers (excluding unknown if desired)
        carriers_to_alert = carrier_summary[carrier_summary["Carrier_Display"] != "UNKNOWN CARRIER"]
        unknown_count = len(critical_df[critical_df["Carrier_Display"] == "UNKNOWN CARRIER"])

        st.info(f"This will send alerts to **{len(carriers_to_alert)}** carriers covering **{critical_count - unknown_count}** shipments")

        if unknown_count > 0:
            st.warning(f"Note: {unknown_count} shipments with unknown carrier will be skipped")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Send Alerts to All Carriers", type="primary", use_container_width=True):
                with st.spinner(f"Sending alerts to {len(carriers_to_alert)} carriers..."):
                    # Prepare data grouped by carrier
                    carriers_data = {}
                    for carrier in carriers_to_alert["Carrier_Display"].tolist():
                        carrier_ships = critical_df[critical_df["Carrier_Display"] == carrier]
                        carriers_data[carrier] = carrier_ships.to_dict('records')

                    results = send_carrier_alerts_webhook(carriers_data)

                # Show results
                successful = [r for r in results if r["success"]]
                failed = [r for r in results if not r["success"]]

                if successful:
                    st.success(f"Successfully sent alerts to {len(successful)} carriers!")
                    for r in successful:
                        st.write(f"- **{r['carrier']}**: {r['shipment_count']} shipments")

                if failed:
                    st.error(f"Failed to send {len(failed)} alerts")
                    for r in failed:
                        st.write(f"- **{r['carrier']}**: {r.get('error', 'Unknown error')}")

                with st.expander("View all webhook payloads"):
                    for r in results:
                        st.markdown(f"### {r['carrier']}")
                        st.json(r["payload"])

    st.divider()

# ==========================================================
# DQ SCORE & SEVERITY DISTRIBUTION (PIE CHARTS)
# ==========================================================
st.header("Data Quality Distribution")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    # DQ Score pie chart
    score_dist = df["score_category"].value_counts().reset_index()
    score_dist.columns = ["Category", "Count"]

    pie_chart(
        score_dist,
        "Count:Q",
        "Category:N",
        "DQ Score Distribution",
        color_scale=SCORE_COLORS
    )

with col2:
    # Severity pie chart
    sev_dist = df["max_severity"].value_counts().reset_index()
    sev_dist.columns = ["Severity", "Count"]

    pie_chart(
        sev_dist,
        "Count:Q",
        "Severity:N",
        "Severity Distribution",
        color_scale=SEVERITY_COLORS
    )

with col3:
    # Delivery status pie chart
    delivery_dist = df["is_delivered"].value_counts().reset_index()
    delivery_dist.columns = ["Status", "Count"]
    delivery_dist["Status"] = delivery_dist["Status"].map({True: "Delivered", False: "In Transit"})

    pie_chart(
        delivery_dist,
        "Count:Q",
        "Status:N",
        "Delivery Status",
        color_scale={"Delivered": "#2ecc71", "In Transit": "#3498db"}
    )

# Summary table below pie charts
st.markdown("**Quick Summary:**")
summary_cols = st.columns(5)
for i, (cat, color) in enumerate(SCORE_COLORS.items()):
    count = (df["score_category"] == cat).sum()
    pct = count / len(df) * 100
    summary_cols[i].markdown(f"<span style='color:{color}; font-weight:bold;'>{cat.split()[0]}</span>: {count:,} ({pct:.1f}%)", unsafe_allow_html=True)

st.divider()

# ==========================================================
# TOP ISSUES AT A GLANCE
# ==========================================================
st.header("Top Issues at a Glance")

# Count rule occurrences
rule_counts = {}
for flags in df["dq_flags"]:
    if flags:
        for rule in flags.split(", "):
            rule = rule.strip()
            if rule:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1

if rule_counts:
    col1, col2 = st.columns([2, 1])

    with col1:
        rule_df = pd.DataFrame([
            {"Rule": k, "Count": v, "Percentage": round(v / len(df) * 100, 1)}
            for k, v in rule_counts.items()
        ]).sort_values("Count", ascending=True)  # ascending for horizontal bar

        horizontal_bar_chart(rule_df, "Count:Q", "Rule:N", "DQ Rule Frequency")

    with col2:
        st.markdown("**Issue Breakdown:**")
        for _, row in rule_df.sort_values("Count", ascending=False).iterrows():
            severity_color = "#e74c3c" if row["Percentage"] > 20 else "#f39c12" if row["Percentage"] > 10 else "#3498db"
            st.markdown(f"- **{row['Rule']}**: {row['Count']:,} ({row['Percentage']}%)")

st.divider()

# ==========================================================
# CARRIER ANALYSIS
# ==========================================================
st.header("Carrier Performance")

carrier_df = df[df["Carrier"].notna()].copy()

if len(carrier_df) > 0:
    # Top carriers by volume
    carrier_stats = (
        carrier_df.groupby("Carrier")
        .agg(
            shipments=("ID", "count"),
            avg_dq_score=("dq_score", "mean"),
            avg_transit=("transit_days", "mean"),
            avg_delay=("delay_days", "mean"),
            critical_pct=("dq_score", lambda x: (x < 60).mean() * 100),
        )
        .reset_index()
        .sort_values("shipments", ascending=False)
        .head(12)
    )

    # Round for display
    carrier_stats["avg_dq_score"] = carrier_stats["avg_dq_score"].round(1)
    carrier_stats["avg_delay"] = carrier_stats["avg_delay"].round(1)
    carrier_stats["critical_pct"] = carrier_stats["critical_pct"].round(1)

    col1, col2 = st.columns(2)

    with col1:
        # Sort for horizontal bar (ascending = True puts highest at top after flip)
        chart_data = carrier_stats.sort_values("avg_dq_score", ascending=True)
        horizontal_bar_chart(chart_data, "avg_dq_score:Q", "Carrier:N", "Avg DQ Score by Carrier")

    with col2:
        chart_data = carrier_stats.sort_values("critical_pct", ascending=True)
        horizontal_bar_chart(chart_data, "critical_pct:Q", "Carrier:N", "Critical Issues % by Carrier")

    # Carrier table
    with st.expander("View Carrier Details Table"):
        st.dataframe(
            carrier_stats[["Carrier", "shipments", "avg_dq_score", "avg_delay", "critical_pct"]]
            .rename(columns={
                "shipments": "Shipments",
                "avg_dq_score": "Avg DQ Score",
                "avg_delay": "Avg Delay (days)",
                "critical_pct": "Critical %"
            }),
            use_container_width=True,
            hide_index=True
        )

st.divider()

# ==========================================================
# LANE ANALYSIS
# ==========================================================
st.header("Lane Analysis")

lane_stats = (
    df.groupby("lane")
    .agg(
        shipments=("ID", "count"),
        avg_dq_score=("dq_score", "mean"),
        critical_pct=("dq_score", lambda x: (x < 60).mean() * 100),
    )
    .reset_index()
    .sort_values("shipments", ascending=False)
    .head(15)
)

lane_stats["avg_dq_score"] = lane_stats["avg_dq_score"].round(1)
lane_stats["critical_pct"] = lane_stats["critical_pct"].round(1)

col1, col2 = st.columns(2)

with col1:
    chart_data = lane_stats.head(10).sort_values("avg_dq_score", ascending=True)
    horizontal_bar_chart(chart_data, "avg_dq_score:Q", "lane:N", "Avg DQ Score by Lane (Top 10)")

with col2:
    # Show lanes with highest critical %
    problem_lanes = lane_stats[lane_stats["critical_pct"] > 0].sort_values("critical_pct", ascending=True).tail(10)
    if len(problem_lanes) > 0:
        horizontal_bar_chart(problem_lanes, "critical_pct:Q", "lane:N", "Lanes with Most Critical Issues")
    else:
        st.info("No lanes with critical issues found.")

st.divider()

# ==========================================================
# SCATTER: TRANSIT vs DQ SCORE
# ==========================================================
st.header("Transit Time vs DQ Score")

scatter_df = df[df["transit_days"].notna() & (df["transit_days"] > -50) & (df["transit_days"] < 150)].copy()
if len(scatter_df) > 1000:
    scatter_df = scatter_df.sample(1000, random_state=42)

if len(scatter_df) > 0:
    scatter_chart(
        scatter_df,
        "transit_days:Q",
        "dq_score:Q",
        "max_severity:N",
        "Transit Days vs DQ Score (colored by severity)",
        color_scale=SEVERITY_COLORS
    )

st.divider()

# ==========================================================
# AGENTIC TRIAGE
# ==========================================================
st.header("Agent Triage & Fix Recommendations")

col1, col2 = st.columns([3, 1])
with col1:
    top_n = st.slider("Number of worst shipments to triage:", 5, 50, 15)
with col2:
    st.markdown("")
    st.markdown("**Agent auto-analyzes worst records**")

with st.spinner("Agent is triaging shipments..."):
    worst = df.sort_values(["dq_score", "max_severity"], ascending=[True, False]).head(top_n).copy()

    triage_results = []
    triage_tickets = []
    triage_objects = {}

    for idx, row in worst.iterrows():
        triage = triage_shipment(row.to_dict())
        ticket = triage["ticket"]
        triage_tickets.append(ticket)
        triage_objects[ticket["shipment_id"]] = triage

        fix = ticket.get("fix_simulation", {}) or {}
        triage_results.append({
            "Shipment ID": ticket["shipment_id"],
            "Carrier": ticket["carrier"] if pd.notna(ticket["carrier"]) else "N/A",
            "Lane": ticket["lane"],
            "DQ Score": ticket["dq_score"],
            "Severity": ticket["severity"],
            "Confidence": ticket["confidence"],
            "Top Rules": ", ".join([d.get("rule", "") for d in ticket.get("triggered_rules", [])[:2]]),
            "Best Fix": fix.get("fix_name", "N/A"),
            "Score After Fix": fix.get("score_after"),
            "Improved": "Yes" if fix.get("improved") else "No",
        })

# Style the dataframe
results_df = pd.DataFrame(triage_results)
st.dataframe(results_df, use_container_width=True, hide_index=True)

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download Triage Tickets (JSON)",
        data=json.dumps(triage_tickets, indent=2, default=str),
        file_name="dq_triage_tickets.json",
        mime="application/json",
    )

st.divider()

# ==========================================================
# DETAILED TRIAGE VIEW
# ==========================================================
st.header("Detailed Triage View")

if triage_tickets:
    chosen_id = st.selectbox(
        "Select a shipment for detailed analysis:",
        options=[t["shipment_id"] for t in triage_tickets],
    )

    chosen_triage = triage_objects.get(chosen_id, {})
    chosen_ticket = next((t for t in triage_tickets if t["shipment_id"] == chosen_id), {})

    if chosen_triage:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        score = chosen_ticket.get("dq_score", 0)
        score_color = "inverse" if score < 60 else "normal" if score < 80 else "off"
        col1.metric("DQ Score", score, delta="Critical" if score < 60 else "OK" if score > 80 else "Fair", delta_color=score_color)
        col2.metric("Severity", chosen_ticket.get("severity"))
        col3.metric("Confidence", f"{chosen_ticket.get('confidence')}")
        col4.metric("Owner", (chosen_ticket.get("suggested_owner", "N/A") or "N/A")[:25])

        # Details in tabs
        tab1, tab2, tab3 = st.tabs(["Observations", "Root Causes & Actions", "Fix Simulation"])

        with tab1:
            for obs in chosen_triage.get("observations", []):
                st.write(f"- {obs}")

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Likely Root Causes:**")
                for cause in chosen_triage.get("likely_root_causes", []):
                    st.write(f"- {cause}")
            with col2:
                st.markdown("**Recommended Actions:**")
                for action in chosen_triage.get("recommended_actions", []):
                    st.write(f"- {action}")

        with tab3:
            fix = chosen_triage.get("fix_simulation", {})
            if fix.get("attempted"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Best Fix", fix.get("fix_name", "N/A")[:30])
                col2.metric("Risk Level", fix.get("risk"))
                col3.metric("Score Improvement", f"{fix.get('score_before')} -> {fix.get('score_after')}")

                if fix.get("candidates_evaluated"):
                    st.markdown("**All Evaluated Fixes:**")
                    for i, cand in enumerate(fix["candidates_evaluated"][:6]):
                        improved = "+" if cand['score_after'] > cand['score_before'] else ""
                        st.write(f"{i+1}. **{cand['fix_name']}** (Risk: {cand['risk']}) - Score: {cand['score_before']} -> {cand['score_after']} {improved}")
            else:
                st.info("No automated fixes available for this shipment.")

st.divider()

# ==========================================================
# SIMULATED ACTIONS
# ==========================================================
st.header("Simulated Actions")

if triage_tickets:
    col1, col2 = st.columns([2, 1])

    with col1:
        chosen_id2 = st.selectbox(
            "Select shipment:",
            options=[t["shipment_id"] for t in triage_tickets],
            key="sim_actions_pick",
        )

    with col2:
        actions = st.multiselect(
            "Actions to simulate:",
            options=["Freshdesk Ticket (simulated)", "Jira Issue (simulated)", "Teams/Email Alert (simulated)"],
            default=["Freshdesk Ticket (simulated)"],
        )

    chosen_ticket2 = next((t for t in triage_tickets if t["shipment_id"] == chosen_id2), {})

    if st.button("Run Simulated Actions", type="primary"):
        with st.spinner("Generating action artifacts..."):
            artifacts = [build_simulated_payload(a, chosen_ticket2) for a in actions]
        st.success(f"Generated {len(artifacts)} simulated action(s)")

        for art in artifacts:
            render_action_card(art)

        st.download_button(
            "Download Action Artifacts (JSON)",
            data=json.dumps(artifacts, indent=2, default=str),
            file_name=f"agent_actions_{chosen_id2}.json",
            mime="application/json",
        )

st.divider()

# ==========================================================
# EXPORT
# ==========================================================
st.header("Export Data")

col1, col2 = st.columns([2, 1])

with col1:
    export_threshold = st.slider("Export shipments with DQ Score below:", 0, 100, 70)

with col2:
    include_outliers = st.checkbox("Include delay outliers", value=True)

if include_outliers:
    export_df = df[(df["dq_score"] < export_threshold) | (df["delay_outlier"])].copy()
else:
    export_df = df[df["dq_score"] < export_threshold].copy()

export_cols = ["ID", "Carrier", "Vessel", "POL", "POD", "lane", "ETD", "ATD", "ETA", "ATA",
               "transit_days", "delay_days", "dq_score", "max_severity", "dq_flags"]
export_cols = [c for c in export_cols if c in export_df.columns]

st.info(f"**{len(export_df):,}** shipments match export criteria")

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "Download Flagged Shipments (CSV)",
        data=export_df[export_cols].to_csv(index=False),
        file_name="flagged_shipments.csv",
        mime="text/csv",
        type="primary"
    )

with col2:
    st.download_button(
        "Download Full Dataset (CSV)",
        data=df[export_cols].to_csv(index=False),
        file_name="all_shipments_scored.csv",
        mime="text/csv",
    )

# Footer
st.divider()
st.markdown("---")
st.caption("TMS Data Quality Analytics Platform V2 - Powered by Agent-driven triage with fix simulation")
