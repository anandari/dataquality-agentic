import pandas as pd
import numpy as np
from datetime import datetime, timezone

IN_FILE = "Final Data Set.csv"
OUT_FILE = "Final Data Set_demo.csv"   # you upload this to Streamlit

RANDOM_SEED = 7

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    df = pd.read_csv(IN_FILE)

    # Parse dates
    for c in ["departure_date", "expected_arrival_date", "actual_arrival_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # Shift all dates +2 years so it aligns with "today" (Feb 2026 demo)
    df["departure_date"] = df["departure_date"] + pd.DateOffset(years=2)
    df["expected_arrival_date"] = df["expected_arrival_date"] + pd.DateOffset(years=2)
    df["actual_arrival_date"] = df["actual_arrival_date"] + pd.DateOffset(years=2)

    # Reference "now" (use real now, or lock to Feb 3, 2026 for stable demos)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
    # now = pd.Timestamp("2026-02-03")  # uncomment to lock demo timing

    n = len(df)

    # ---------------------------
    # 1) Create PLANNED shipments for dashboard sections 2–5
    # ---------------------------
    planned_idx = rng.choice(df.index, size=int(n * 0.18), replace=False)

    # departure_date 1..7 days in the future → ETD countdown window
    df.loc[planned_idx, "departure_date"] = now + pd.to_timedelta(
        rng.integers(1, 8, size=len(planned_idx)), unit="D"
    )

    # expected arrival = departure + transit time
    df["transit_time_days"] = pd.to_numeric(df["transit_time_days"], errors="coerce")
    df.loc[planned_idx, "expected_arrival_date"] = (
        df.loc[planned_idx, "departure_date"]
        + pd.to_timedelta(df.loc[planned_idx, "transit_time_days"].fillna(20), unit="D")
    )

    # planned → no actual yet
    df.loc[planned_idx, "actual_arrival_date"] = pd.NaT

    # ---------------------------
    # 2) Force meaningful risk variation (so risk list is not empty)
    # ---------------------------
    carriers = df["carrier"].dropna().unique().tolist()
    if len(carriers) < 2:
        bad_carriers = carriers
    else:
        bad_carriers = carriers[:2]  # first 2 carriers become "riskier patterns"

    # Choose a high-volume origin port + top destinations to create a "bad lane cluster"
    top_origin = df["origin_port"].value_counts().index[0]
    top_dests = df["destination_port"].value_counts().head(3).index.tolist()
    lane_mask = (df["origin_port"] == top_origin) & (df["destination_port"].isin(top_dests))

    # Peak season months
    peak_mask = df["departure_date"].dt.month.isin([11, 12])

    # Combine patterns: bad carriers in peak season OR bad lanes
    pattern_idx = df.index[(df["carrier"].isin(bad_carriers) & peak_mask) | lane_mask]
    sel_bad = rng.choice(pattern_idx, size=min(len(pattern_idx), int(n * 0.20)), replace=False)

    # Inject worse operational conditions (creates visible chart differences)
    df.loc[sel_bad, "route_congestion_index"] = rng.normal(8.2, 1.0, size=len(sel_bad)).clip(5, 10)
    df.loc[sel_bad, "carrier_historical_delay_rate"] = rng.normal(0.55, 0.12, size=len(sel_bad)).clip(0.2, 0.9)
    df.loc[sel_bad, "eta_revision_count"] = rng.integers(3, 7, size=len(sel_bad))
    df.loc[sel_bad, "milestone_update_lag_hours"] = rng.normal(55, 20, size=len(sel_bad)).clip(24, 120)

    # Missing fields increased → lower data completeness → higher risk
    df.loc[sel_bad, "mandatory_fields_missing_count"] = rng.integers(2, 7, size=len(sel_bad))
    filled = pd.to_numeric(df["mandatory_fields_filled_count"], errors="coerce").fillna(20)
    df.loc[sel_bad, "mandatory_fields_filled_count"] = (filled.loc[sel_bad] - rng.integers(0, 3, size=len(sel_bad))).clip(5, 30)

    # ---------------------------
    # 3) Create delivery delays on delivered shipments (so outliers + delay charts have contrast)
    # ---------------------------
    delivered_idx = df.index[df["actual_arrival_date"].notna()]
    sel_delayed = rng.choice(delivered_idx, size=int(n * 0.15), replace=False)
    df.loc[sel_delayed, "actual_arrival_date"] = (
        df.loc[sel_delayed, "expected_arrival_date"]
        + pd.to_timedelta(rng.integers(1, 10, size=len(sel_delayed)), unit="D")
    )

    # ---------------------------
    # 4) Clean up null congestion (keep variety)
    # ---------------------------
    df["route_congestion_index"] = pd.to_numeric(df["route_congestion_index"], errors="coerce")
    null_cong = df["route_congestion_index"].isna()
    if null_cong.any():
        df.loc[null_cong, "route_congestion_index"] = rng.uniform(0, 10, size=int(null_cong.sum()))

    df.to_csv(OUT_FILE, index=False)
    print(f"✅ Wrote demo dataset: {OUT_FILE}")
    print(f"Planned shipments (%): {(df['actual_arrival_date'].isna().mean()*100):.1f}%")

if __name__ == "__main__":
    main()
