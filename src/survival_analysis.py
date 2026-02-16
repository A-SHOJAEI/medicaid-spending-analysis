"""Phase 8A: Provider Survival Analysis.

Models provider "survival" — time until a provider becomes inactive or drops
out of Medicaid billing — using clinical survival analysis methods.

Methods:
    - Kaplan-Meier estimator by provider size tier
    - Cox Proportional Hazards regression
    - Log-rank tests between groups
    - Hazard ratio forest plot
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table,
)

logger = setup_logging()


# ── Data Preparation ─────────────────────────────────────────────────


def build_survival_data(config: dict) -> pd.DataFrame:
    """Build a survival dataset from yearly parquet files.

    For each provider, compute:
        - duration: number of months from first to last observed activity
        - event: 1 if the provider was NOT active in the most recent year (2024),
                 0 if still active (right-censored)
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    logger.info("Building provider survival dataset...")

    # Collect first/last month per provider across all years
    records = []
    for year in range(2018, 2025):
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(
            path, columns=["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH",
                           "TOTAL_CLAIMS", "TOTAL_PAID"]
        )
        agg = df.groupby("BILLING_PROVIDER_NPI_NUM").agg(
            min_month=("CLAIM_FROM_MONTH", "min"),
            max_month=("CLAIM_FROM_MONTH", "max"),
            total_claims=("TOTAL_CLAIMS", "sum"),
            total_paid=("TOTAL_PAID", "sum"),
        ).reset_index()
        agg["year"] = year
        records.append(agg)
        del df
        gc.collect()

    all_activity = pd.concat(records, ignore_index=True)

    # Aggregate per provider: first ever month, last ever month
    provider_agg = all_activity.groupby("BILLING_PROVIDER_NPI_NUM").agg(
        first_month=("min_month", "min"),
        last_month=("max_month", "max"),
        total_claims=("total_claims", "sum"),
        total_paid=("total_paid", "sum"),
        n_years_active=("year", "nunique"),
    ).reset_index()

    # Parse months to compute duration
    provider_agg["first_date"] = pd.to_datetime(provider_agg["first_month"], format="%Y-%m")
    provider_agg["last_date"] = pd.to_datetime(provider_agg["last_month"], format="%Y-%m")
    provider_agg["duration_months"] = (
        (provider_agg["last_date"].dt.year - provider_agg["first_date"].dt.year) * 12
        + (provider_agg["last_date"].dt.month - provider_agg["first_date"].dt.month)
        + 1  # inclusive
    )

    # Determine the latest observed month in the dataset
    global_last = provider_agg["last_date"].max()
    # Right-censoring cutoff: consider 2024-06 or later as "still active"
    cutoff = global_last - pd.DateOffset(months=6)
    provider_agg["event"] = (provider_agg["last_date"] < cutoff).astype(int)

    # Provider size tier
    paid_q = provider_agg["total_paid"].quantile([0.5, 0.9, 0.99])
    conditions = [
        provider_agg["total_paid"] < paid_q.iloc[0],
        provider_agg["total_paid"] < paid_q.iloc[1],
        provider_agg["total_paid"] < paid_q.iloc[2],
        provider_agg["total_paid"] >= paid_q.iloc[2],
    ]
    labels = ["Small (<p50)", "Medium (p50-p90)", "Large (p90-p99)", "Mega (>p99)"]
    provider_agg["size_tier"] = np.select(conditions, labels, default="Unknown")

    # Claims per month as intensity
    provider_agg["claims_per_month"] = provider_agg["total_claims"] / provider_agg["duration_months"]

    # Merge provider features
    feat_path = processed_dir / "provider_features.parquet"
    if feat_path.exists():
        feats = pd.read_parquet(feat_path)
        provider_agg = provider_agg.merge(
            feats[["billing_npi", "n_unique_hcpcs", "n_servicing_npis",
                   "paid_per_claim", "claims_per_beneficiary"]],
            left_on="BILLING_PROVIDER_NPI_NUM",
            right_on="billing_npi",
            how="left",
        )

    logger.info(f"Survival dataset: {len(provider_agg)} providers, "
                f"event rate = {provider_agg['event'].mean():.1%}")
    return provider_agg


# ── Kaplan-Meier ─────────────────────────────────────────────────────


def fit_kaplan_meier(surv_df: pd.DataFrame, config: dict) -> dict:
    """Fit Kaplan-Meier estimators by size tier and generate plots."""
    setup_plotting(config)

    tiers = ["Small (<p50)", "Medium (p50-p90)", "Large (p90-p99)", "Mega (>p99)"]
    colors = {"Small (<p50)": "#2196F3", "Medium (p50-p90)": "#4CAF50",
              "Large (p90-p99)": "#FF9800", "Mega (>p99)": "#F44336"}

    # ── Main KM plot ──
    fig, ax = plt.subplots(figsize=(12, 8))
    km_results = {}

    for tier in tiers:
        mask = surv_df["size_tier"] == tier
        if mask.sum() < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            surv_df.loc[mask, "duration_months"],
            event_observed=surv_df.loc[mask, "event"],
            label=tier,
        )
        kmf.plot_survival_function(ax=ax, color=colors.get(tier, "gray"), ci_show=True)
        km_results[tier] = {
            "n": int(mask.sum()),
            "n_events": int(surv_df.loc[mask, "event"].sum()),
            "median_survival": float(kmf.median_survival_time_)
            if not np.isinf(kmf.median_survival_time_) else None,
            "survival_12m": float(kmf.predict(12)),
            "survival_24m": float(kmf.predict(24)),
            "survival_48m": float(kmf.predict(48)),
        }

    ax.set_xlabel("Duration (months)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Provider Survival by Size Tier (Kaplan-Meier)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left")
    ax.axhline(y=0.5, color="gray", ls=":", alpha=0.5)
    save_figure(fig, "survival_kaplan_meier", config)

    # ── KM by years active ──
    surv_df["activity_group"] = pd.cut(
        surv_df["n_years_active"],
        bins=[0, 1, 3, 5, 7],
        labels=["1 year", "2-3 years", "4-5 years", "6-7 years"],
    )
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    activity_colors = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"]
    for i, grp in enumerate(["1 year", "2-3 years", "4-5 years", "6-7 years"]):
        mask = surv_df["activity_group"] == grp
        if mask.sum() < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(surv_df.loc[mask, "duration_months"],
                event_observed=surv_df.loc[mask, "event"], label=grp)
        kmf.plot_survival_function(ax=ax2, color=activity_colors[i], ci_show=True)
    ax2.set_xlabel("Duration (months)")
    ax2.set_ylabel("Survival Probability")
    ax2.set_title("Provider Survival by Activity Span",
                  fontsize=14, fontweight="bold")
    ax2.legend(loc="lower left")
    save_figure(fig2, "survival_by_activity", config)

    return km_results


# ── Log-Rank Tests ───────────────────────────────────────────────────


def run_logrank_tests(surv_df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise log-rank tests between size tiers."""
    tiers = [t for t in ["Small (<p50)", "Medium (p50-p90)",
                         "Large (p90-p99)", "Mega (>p99)"]
             if (surv_df["size_tier"] == t).sum() >= 10]
    rows = []

    # Overall test
    multi_result = multivariate_logrank_test(
        surv_df["duration_months"],
        surv_df["size_tier"],
        surv_df["event"],
    )
    rows.append({
        "group_1": "ALL",
        "group_2": "ALL",
        "test_statistic": float(multi_result.test_statistic),
        "p_value": float(multi_result.p_value),
    })

    # Pairwise
    for i, t1 in enumerate(tiers):
        for t2 in tiers[i + 1:]:
            m1 = surv_df["size_tier"] == t1
            m2 = surv_df["size_tier"] == t2
            result = logrank_test(
                surv_df.loc[m1, "duration_months"],
                surv_df.loc[m2, "duration_months"],
                event_observed_A=surv_df.loc[m1, "event"],
                event_observed_B=surv_df.loc[m2, "event"],
            )
            rows.append({
                "group_1": t1,
                "group_2": t2,
                "test_statistic": float(result.test_statistic),
                "p_value": float(result.p_value),
            })

    lr_df = pd.DataFrame(rows)
    logger.info(f"Log-rank tests:\n{lr_df.to_string()}")
    return lr_df


# ── Cox Proportional Hazards ─────────────────────────────────────────


def fit_cox_model(surv_df: pd.DataFrame, config: dict) -> dict:
    """Fit Cox PH model and generate coefficient/hazard plots."""
    setup_plotting(config)

    cox_features = ["claims_per_month", "n_years_active"]
    optional = ["n_unique_hcpcs", "n_servicing_npis", "paid_per_claim",
                "claims_per_beneficiary"]
    for col in optional:
        if col in surv_df.columns:
            cox_features.append(col)

    cox_df = surv_df[["duration_months", "event"] + cox_features].copy()
    cox_df = cox_df.replace([np.inf, -np.inf], np.nan).dropna()
    # Log-transform skewed features
    for col in ["claims_per_month", "paid_per_claim"]:
        if col in cox_df.columns:
            cox_df[col] = np.log1p(cox_df[col].clip(lower=0))

    # Standardize
    for col in cox_features:
        if col in cox_df.columns:
            mu, sigma = cox_df[col].mean(), cox_df[col].std()
            if sigma > 0:
                cox_df[col] = (cox_df[col] - mu) / sigma

    logger.info(f"Cox model input: {len(cox_df)} rows, {cox_df.isnull().sum().sum()} NaNs")

    # Fit model
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_df, duration_col="duration_months", event_col="event")
    logger.info(f"Cox PH summary:\n{cph.summary.to_string()}")

    # ── Hazard ratio forest plot ──
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = cph.summary
    coefs = summary["coef"]
    ci_lower = summary["coef lower 95%"]
    ci_upper = summary["coef upper 95%"]
    hr = np.exp(coefs)
    hr_lower = np.exp(ci_lower)
    hr_upper = np.exp(ci_upper)

    y_pos = range(len(coefs))
    ax.errorbar(hr.values, y_pos,
                xerr=[hr.values - hr_lower.values, hr_upper.values - hr.values],
                fmt="o", color="#2196F3", capsize=5, capthick=2, markersize=8)
    ax.axvline(x=1.0, color="red", ls="--", alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(coefs.index)
    ax.set_xlabel("Hazard Ratio (exp(coef))")
    ax.set_title("Cox PH Hazard Ratios (95% CI)", fontsize=14, fontweight="bold")
    save_figure(fig, "survival_hazard_ratios", config)

    # ── Cox coefficients bar plot ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ["#F44336" if c > 0 else "#4CAF50" for c in coefs.values]
    ax2.barh(list(y_pos), coefs.values, color=colors, alpha=0.8)
    ax2.axvline(x=0, color="gray", ls="-", alpha=0.5)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(coefs.index)
    ax2.set_xlabel("Cox Coefficient (log hazard ratio)")
    ax2.set_title("Cox PH Coefficients", fontsize=14, fontweight="bold")
    for i, (v, p) in enumerate(zip(coefs.values, summary["p"].values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax2.text(v + 0.01 * np.sign(v), i, sig, va="center", fontweight="bold")
    save_figure(fig2, "survival_cox_coefficients", config)

    # ── Survival function at different covariate levels ──
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    # Predict at quartiles of claims_per_month
    if "claims_per_month" in cox_df.columns:
        median_row = cox_df[cox_features].median()
        for q, label, color in [(0.1, "Low billing (p10)", "#4CAF50"),
                                 (0.5, "Median billing (p50)", "#2196F3"),
                                 (0.9, "High billing (p90)", "#F44336")]:
            row = median_row.copy()
            row["claims_per_month"] = cox_df["claims_per_month"].quantile(q)
            sf = cph.predict_survival_function(row.to_frame().T)
            ax3.plot(sf.index, sf.values.flatten(), label=label, color=color, lw=2)
        ax3.set_xlabel("Duration (months)")
        ax3.set_ylabel("Survival Probability")
        ax3.set_title("Predicted Survival at Different Billing Intensities",
                      fontsize=14, fontweight="bold")
        ax3.legend()
        save_figure(fig3, "survival_predicted_curves", config)
    else:
        plt.close(fig3)

    cox_results = {
        "concordance": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_ratio_test().test_statistic),
        "log_likelihood_p": float(cph.log_likelihood_ratio_test().p_value),
        "n_observations": int(len(cox_df)),
        "n_events": int(cox_df["event"].sum()),
        "coefficients": {k: float(v) for k, v in coefs.items()},
        "hazard_ratios": {k: float(v) for k, v in hr.items()},
        "p_values": {k: float(v) for k, v in summary["p"].items()},
    }
    return cox_results


# ── Event Rate Analysis ──────────────────────────────────────────────


def plot_event_rate_analysis(surv_df: pd.DataFrame, config: dict) -> dict:
    """Analyze dropout rates by cohort entry year."""
    setup_plotting(config)

    surv_df["entry_year"] = surv_df["first_date"].dt.year
    cohort_stats = surv_df.groupby("entry_year").agg(
        n_providers=("BILLING_PROVIDER_NPI_NUM", "count"),
        event_rate=("event", "mean"),
        median_duration=("duration_months", "median"),
        mean_duration=("duration_months", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(cohort_stats["entry_year"], cohort_stats["n_providers"],
                color="#2196F3", alpha=0.8)
    axes[0].set_xlabel("Entry Year")
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_title("New Provider Entries by Year")

    axes[1].plot(cohort_stats["entry_year"], cohort_stats["event_rate"],
                 "o-", color="#F44336", lw=2)
    axes[1].set_xlabel("Entry Year")
    axes[1].set_ylabel("Dropout Rate")
    axes[1].set_title("Provider Dropout Rate by Entry Cohort")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    axes[2].bar(cohort_stats["entry_year"], cohort_stats["median_duration"],
                color="#4CAF50", alpha=0.8)
    axes[2].set_xlabel("Entry Year")
    axes[2].set_ylabel("Median Duration (months)")
    axes[2].set_title("Median Active Duration by Entry Cohort")

    fig.suptitle("Provider Cohort Analysis", fontsize=14, fontweight="bold")
    save_figure(fig, "survival_cohort_analysis", config)

    return {
        "n_cohorts": len(cohort_stats),
        "highest_dropout_year": int(cohort_stats.loc[
            cohort_stats["event_rate"].idxmax(), "entry_year"]),
        "highest_dropout_rate": float(cohort_stats["event_rate"].max()),
    }


# ── Main Pipeline ────────────────────────────────────────────────────


def run_survival_analysis(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]

    # Build survival dataset
    surv_df = build_survival_data(config)
    surv_df.to_parquet(processed_dir / "provider_survival.parquet", index=False)
    save_table(surv_df.describe().T, "survival_descriptive_stats", config)

    # Kaplan-Meier
    logger.info("=== Fitting Kaplan-Meier estimators ===")
    km_results = fit_kaplan_meier(surv_df, config)

    # Log-rank tests
    logger.info("=== Running log-rank tests ===")
    lr_df = run_logrank_tests(surv_df)
    save_table(lr_df, "survival_logrank_tests", config)

    # Cox PH
    logger.info("=== Fitting Cox Proportional Hazards model ===")
    cox_results = fit_cox_model(surv_df, config)

    # Cohort analysis
    logger.info("=== Cohort dropout analysis ===")
    cohort_results = plot_event_rate_analysis(surv_df, config)

    # Summary
    summary = {
        "n_providers": len(surv_df),
        "event_rate": float(surv_df["event"].mean()),
        "median_duration_months": float(surv_df["duration_months"].median()),
        "mean_duration_months": float(surv_df["duration_months"].mean()),
        "kaplan_meier": km_results,
        "cox_ph": cox_results,
        "cohort_analysis": cohort_results,
    }

    with open(tables_dir / "survival_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Survival summary: concordance={cox_results['concordance']:.3f}, "
                f"event_rate={summary['event_rate']:.1%}")

    gc.collect()
    logger.info("Phase 8A: Survival Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8A: Survival Analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_survival_analysis(config)
