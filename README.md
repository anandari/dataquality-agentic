📦 TMS Data Quality Analytics Platform

🧱 Technology Stack & Intelligence Model
🐍 Python 3
Core language — all logic, scoring, simulation runs locally.

📊 Pandas & NumPy
Data processing, aggregations, delay calculations, risk scoring math.

🎛️ Streamlit
Interactive dashboard UI — file upload, filters, tables, exports.

📉 Altair
Clear visual analytics — heatmaps, trends, comparisons with readable axes.

✅ YAML-based DQ rule engine
Explainable data quality rules with penalties and severities.

🧠 Local risk scoring & explanations
Deterministic scoring using data completeness, congestion, carrier history, ETA churn.

🤖 Agentic triage & simulated orchestration
A rule-driven agent that observes, scores, simulates fixes, and recommends actions
(Jira / Freshdesk / Email are simulated — no real write-back).

------------------------------------------------------------------------------------------------

pip install pandas numpy streamlit
➡️ Installs the core data processing libraries and the dashboard framework.

python3 -m pip install --upgrade pip
➡️ Ensures the Python package manager is up to date and stable.

python3 -c "import pandas, numpy, streamlit; print('ALL GOOD')"
➡️ Quick sanity check that the main libraries are installed correctly.

python3 -m pip install pyyaml
➡️ Enables loading human-readable data quality rules from YAML files.

python3 -m pip install altair
➡️ Installs the charting library used for heatmaps, trends, and comparisons.

➡️ To run use:
streamlit run app.py

------------------------------------------------------------------------------------------------
