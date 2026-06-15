"""
Customer Churn Analysis Pipeline
=================================
Analyzes subscription customer data to identify patterns behind cancellations.
Evaluates engagement, activity, and behavioral trends to surface key churn drivers
and generate actionable retention recommendations.

Usage:
    python scripts/churn_analysis.py
    python scripts/churn_analysis.py --input data/raw/customers_raw.csv
"""

import pandas as pd
import numpy as np
import argparse
import logging
import json
from io import StringIO
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_data(path: str | None) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        log.info(f"Loaded {len(df)} rows from '{path}'")
    else:
        from io import StringIO
        import os
        default = "data/raw/customers_raw.csv"
        if os.path.exists(default):
            df = pd.read_csv(default)
            log.info(f"Loaded {len(df)} rows from '{default}'")
        else:
            raise FileNotFoundError("No input file found. Pass --input <path>")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive behavioral and value metrics from raw columns."""

    # Engagement score (0–100): composite of logins, recency, support dependency
    df["engagement_score"] = (
        (df["login_frequency"].clip(0, 30) / 30 * 40)       # 40pts: login activity
        + ((90 - df["last_active_days"].clip(0, 90)) / 90 * 40)  # 40pts: recency
        + ((10 - df["support_calls"].clip(0, 10)) / 10 * 20)     # 20pts: low friction
    ).round(1)

    # Customer lifetime value proxy
    df["ltv_estimate"] = (df["monthly_charge"] * df["tenure_months"]).round(2)

    # Revenue risk: monthly charge × churn probability (rough flag)
    df["high_value"] = df["monthly_charge"] >= 59.99

    # Tenure bands
    df["tenure_band"] = pd.cut(
        df["tenure_months"],
        bins=[0, 3, 12, 24, 48, 999],
        labels=["0–3 mo", "4–12 mo", "13–24 mo", "25–48 mo", "48+ mo"],
    )

    # NPS sentiment band
    df["nps_segment"] = pd.cut(
        df["nps_score"],
        bins=[0, 6, 8, 10],
        labels=["Detractor", "Passive", "Promoter"],
        include_lowest=True,
    )

    # At-risk flag: low engagement + high support calls + short tenure
    df["at_risk"] = (
        (df["engagement_score"] < 30)
        & (df["support_calls"] >= 5)
        & (df["tenure_months"] <= 12)
    ).astype(int)

    log.info("Feature engineering complete")
    return df


# ─────────────────────────────────────────────────────────────
# 3. CORE CHURN METRICS
# ─────────────────────────────────────────────────────────────

def compute_churn_metrics(df: pd.DataFrame) -> dict:
    total        = len(df)
    churned      = df["churned"].sum()
    retained     = total - churned
    churn_rate   = churned / total * 100

    avg_tenure_churned  = df[df["churned"] == 1]["tenure_months"].mean()
    avg_tenure_retained = df[df["churned"] == 0]["tenure_months"].mean()

    revenue_at_risk = df[df["churned"] == 1]["monthly_charge"].sum()
    avg_ltv_churned  = df[df["churned"] == 1]["ltv_estimate"].mean()
    avg_ltv_retained = df[df["churned"] == 0]["ltv_estimate"].mean()

    return {
        "total_customers": int(total),
        "churned":         int(churned),
        "retained":        int(retained),
        "churn_rate_pct":  round(churn_rate, 2),
        "avg_tenure_churned_months":  round(avg_tenure_churned, 1),
        "avg_tenure_retained_months": round(avg_tenure_retained, 1),
        "monthly_revenue_at_risk":    round(revenue_at_risk, 2),
        "avg_ltv_churned":   round(avg_ltv_churned, 2),
        "avg_ltv_retained":  round(avg_ltv_retained, 2),
    }


# ─────────────────────────────────────────────────────────────
# 4. SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────

def churn_by_segment(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Churn rate broken down by any categorical column."""
    grp = df.groupby(col)["churned"].agg(["sum", "count"]).reset_index()
    grp.columns = [col, "churned", "total"]
    grp["churn_rate_pct"] = (grp["churned"] / grp["total"] * 100).round(1)
    grp["retained"] = grp["total"] - grp["churned"]
    return grp.sort_values("churn_rate_pct", ascending=False)


def engagement_analysis(df: pd.DataFrame) -> dict:
    churned  = df[df["churned"] == 1]
    retained = df[df["churned"] == 0]

    return {
        "avg_engagement_churned":  round(churned["engagement_score"].mean(), 1),
        "avg_engagement_retained": round(retained["engagement_score"].mean(), 1),
        "avg_logins_churned":      round(churned["login_frequency"].mean(), 1),
        "avg_logins_retained":     round(retained["login_frequency"].mean(), 1),
        "avg_support_calls_churned":  round(churned["support_calls"].mean(), 1),
        "avg_support_calls_retained": round(retained["support_calls"].mean(), 1),
        "avg_days_inactive_churned":  round(churned["last_active_days"].mean(), 1),
        "avg_days_inactive_retained": round(retained["last_active_days"].mean(), 1),
        "avg_nps_churned":   round(churned["nps_score"].mean(), 1),
        "avg_nps_retained":  round(retained["nps_score"].mean(), 1),
    }


# ─────────────────────────────────────────────────────────────
# 5. CHURN DRIVERS  (correlation + mean-difference ranking)
# ─────────────────────────────────────────────────────────────

def rank_churn_drivers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        "tenure_months", "monthly_charge", "support_calls",
        "login_frequency", "last_active_days", "nps_score",
        "num_products", "engagement_score",
    ]
    rows = []
    for feat in numeric_features:
        churned_mean  = df[df["churned"] == 1][feat].mean()
        retained_mean = df[df["churned"] == 0][feat].mean()
        corr = df[feat].corr(df["churned"])
        rows.append({
            "feature":        feat,
            "churned_mean":   round(churned_mean, 2),
            "retained_mean":  round(retained_mean, 2),
            "difference":     round(churned_mean - retained_mean, 2),
            "correlation_with_churn": round(corr, 3),
        })
    drivers = pd.DataFrame(rows)
    drivers["abs_corr"] = drivers["correlation_with_churn"].abs()
    return drivers.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")


# ─────────────────────────────────────────────────────────────
# 6. COHORT RETENTION TABLE
# ─────────────────────────────────────────────────────────────

def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("tenure_band")
        .agg(
            total=("churned", "count"),
            churned=("churned", "sum"),
        )
        .assign(retention_rate=lambda x: ((x["total"] - x["churned"]) / x["total"] * 100).round(1))
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────
# 7. RETENTION RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

def generate_recommendations(metrics: dict, engagement: dict, drivers: pd.DataFrame) -> list[dict]:
    recs = []

    # Short tenure churn
    if metrics["avg_tenure_churned_months"] < 6:
        recs.append({
            "priority": "HIGH",
            "finding":  "Most churn occurs in the first 6 months",
            "action":   "Launch a 90-day onboarding journey: welcome emails, feature spotlights, and a check-in call at day 30.",
            "segment":  "New customers (0–6 months)",
        })

    # Low engagement
    if engagement["avg_engagement_churned"] < 35:
        recs.append({
            "priority": "HIGH",
            "finding":  f"Churned users have avg engagement score of {engagement['avg_engagement_churned']} vs {engagement['avg_engagement_retained']} for retained",
            "action":   "Trigger re-engagement emails when a user hasn't logged in for 14 days. Offer a feature tutorial or incentive.",
            "segment":  "Low-engagement users",
        })

    # High support calls
    if engagement["avg_support_calls_churned"] > 5:
        recs.append({
            "priority": "HIGH",
            "finding":  f"Churned customers averaged {engagement['avg_support_calls_churned']} support calls vs {engagement['avg_support_calls_retained']} for retained",
            "action":   "Flag accounts with 3+ support calls in 30 days for proactive CSM outreach. Investigate top support topics for product fixes.",
            "segment":  "High-friction customers",
        })

    # NPS detractors
    if engagement["avg_nps_churned"] < 4:
        recs.append({
            "priority": "MEDIUM",
            "finding":  f"Churned customers had avg NPS of {engagement['avg_nps_churned']} — well below promoter threshold",
            "action":   "Run monthly NPS surveys. Route detractors (score ≤ 6) to a dedicated retention specialist within 48 hours.",
            "segment":  "NPS Detractors",
        })

    # Contract type
    recs.append({
        "priority": "MEDIUM",
        "finding":  "Month-to-month customers churn at significantly higher rates than annual/biennial contract holders",
        "action":   "Offer 15–20% discount for customers who upgrade from monthly to annual plans. Promote during month 2–3.",
        "segment":  "Month-to-month subscribers",
    })

    # Revenue risk
    if metrics["monthly_revenue_at_risk"] > 0:
        recs.append({
            "priority": "HIGH",
            "finding":  f"${metrics['monthly_revenue_at_risk']:,.2f}/month in MRR at risk from churned customers",
            "action":   "Identify at-risk high-value accounts (Premium plan + engagement score < 40) and assign dedicated account managers.",
            "segment":  "High-value at-risk accounts",
        })

    # Low product adoption
    recs.append({
        "priority": "LOW",
        "finding":  "Churned customers use fewer products/features on average",
        "action":   "Build in-app product adoption nudges. Surface unused features contextually. Track feature adoption as a leading churn indicator.",
        "segment":  "Low product adoption users",
    })

    return recs


# ─────────────────────────────────────────────────────────────
# 8. REPORT GENERATION
# ─────────────────────────────────────────────────────────────

def print_report(metrics: dict, engagement: dict, drivers: pd.DataFrame,
                 plan_seg: pd.DataFrame, contract_seg: pd.DataFrame,
                 cohort: pd.DataFrame, recs: list[dict]) -> None:

    sep = "═" * 60

    print(f"\n{sep}")
    print("  CUSTOMER CHURN ANALYSIS REPORT")
    print(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(sep)

    print("\n── HEADLINE METRICS ─────────────────────────────────────")
    print(f"  Total customers       : {metrics['total_customers']}")
    print(f"  Churned               : {metrics['churned']} ({metrics['churn_rate_pct']}%)")
    print(f"  Retained              : {metrics['retained']}")
    print(f"  Avg tenure (churned)  : {metrics['avg_tenure_churned_months']} months")
    print(f"  Avg tenure (retained) : {metrics['avg_tenure_retained_months']} months")
    print(f"  MRR at risk           : ${metrics['monthly_revenue_at_risk']:,.2f}")
    print(f"  Avg LTV (churned)     : ${metrics['avg_ltv_churned']:,.2f}")
    print(f"  Avg LTV (retained)    : ${metrics['avg_ltv_retained']:,.2f}")

    print("\n── ENGAGEMENT COMPARISON ────────────────────────────────")
    print(f"  {'Metric':<30} {'Churned':>10} {'Retained':>10}")
    print(f"  {'-'*50}")
    eng_rows = [
        ("Engagement score",    engagement['avg_engagement_churned'],  engagement['avg_engagement_retained']),
        ("Login frequency",     engagement['avg_logins_churned'],      engagement['avg_logins_retained']),
        ("Support calls",       engagement['avg_support_calls_churned'], engagement['avg_support_calls_retained']),
        ("Days since active",   engagement['avg_days_inactive_churned'], engagement['avg_days_inactive_retained']),
        ("NPS score",           engagement['avg_nps_churned'],         engagement['avg_nps_retained']),
    ]
    for label, c, r in eng_rows:
        print(f"  {label:<30} {c:>10.1f} {r:>10.1f}")

    print("\n── CHURN DRIVERS (ranked by correlation) ────────────────")
    print(f"  {'Feature':<25} {'Churned':>10} {'Retained':>10} {'Correlation':>13}")
    print(f"  {'-'*60}")
    for _, row in drivers.iterrows():
        print(f"  {row['feature']:<25} {row['churned_mean']:>10.2f} {row['retained_mean']:>10.2f} {row['correlation_with_churn']:>13.3f}")

    print("\n── CHURN BY PLAN ────────────────────────────────────────")
    for _, row in plan_seg.iterrows():
        bar = "█" * int(row["churn_rate_pct"] / 2)
        print(f"  {row['plan']:<12} {row['churn_rate_pct']:>5.1f}%  {bar}")

    print("\n── CHURN BY CONTRACT TYPE ───────────────────────────────")
    for _, row in contract_seg.iterrows():
        bar = "█" * int(row["churn_rate_pct"] / 2)
        print(f"  {row['contract_type']:<22} {row['churn_rate_pct']:>5.1f}%  {bar}")

    print("\n── COHORT RETENTION ─────────────────────────────────────")
    print(f"  {'Tenure Band':<14} {'Total':>7} {'Churned':>9} {'Retention':>11}")
    print(f"  {'-'*43}")
    for _, row in cohort.iterrows():
        print(f"  {str(row['tenure_band']):<14} {row['total']:>7} {row['churned']:>9} {row['retention_rate']:>10.1f}%")

    print("\n── RETENTION RECOMMENDATIONS ────────────────────────────")
    for i, rec in enumerate(recs, 1):
        print(f"\n  [{rec['priority']}] #{i} — {rec['segment']}")
        print(f"  Finding : {rec['finding']}")
        print(f"  Action  : {rec['action']}")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────
# 9. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, metrics: dict, drivers: pd.DataFrame,
                 recs: list[dict]) -> None:

    # Enriched customer data
    df.to_csv(OUTPUT_DIR / "customers_enriched.csv", index=False)
    log.info(f"Saved enriched dataset → {OUTPUT_DIR}/customers_enriched.csv")

    # At-risk customers
    at_risk = df[(df["at_risk"] == 1) & (df["churned"] == 0)]
    at_risk.to_csv(OUTPUT_DIR / "at_risk_customers.csv", index=False)
    log.info(f"Saved {len(at_risk)} at-risk customers → {OUTPUT_DIR}/at_risk_customers.csv")

    # Churn drivers
    drivers.to_csv(OUTPUT_DIR / "churn_drivers.csv", index=False)

    # Recommendations JSON
    with open(OUTPUT_DIR / "recommendations.json", "w") as f:
        json.dump(recs, f, indent=2)

    # Summary metrics JSON
    with open(OUTPUT_DIR / "summary_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"All outputs saved to ./{OUTPUT_DIR}/")


# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────

def run(input_path: str | None = None) -> None:
    df = load_data(input_path)
    df = engineer_features(df)

    metrics      = compute_churn_metrics(df)
    engagement   = engagement_analysis(df)
    drivers      = rank_churn_drivers(df)
    plan_seg     = churn_by_segment(df, "plan")
    contract_seg = churn_by_segment(df, "contract_type")
    cohort       = cohort_retention(df)
    recs         = generate_recommendations(metrics, engagement, drivers)

    print_report(metrics, engagement, drivers, plan_seg, contract_seg, cohort, recs)
    save_outputs(df, metrics, drivers, recs)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Churn Analysis Pipeline")
    parser.add_argument("--input", "-i", default=None, help="Path to raw customer CSV")
    args = parser.parse_args()
    run(args.input)
