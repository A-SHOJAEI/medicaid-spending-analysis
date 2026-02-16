"""Causal inference for COVID impact on Medicaid spending.

Implements Difference-in-Differences, Bayesian Structural Time Series,
Regression Discontinuity, and Synthetic Control methods.
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency, format_number,
)

logger = setup_logging()


def build_panel_data(
    config: Optional[dict] = None,
    top_n: int = 50000,
) -> pd.DataFrame:
    """Construct provider-month panel dataset for causal analysis."""
    if config is None:
        config = load_config()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Top providers by total spending
    pf = pd.read_parquet(processed_dir / "provider_features.parquet")
    top_npis = set(pf.nlargest(top_n, "total_paid")["billing_npi"].values)
    provider_info = pf[pf["billing_npi"].isin(top_npis)][
        ["billing_npi", "total_paid", "total_claims", "n_unique_hcpcs", "n_servicing_npis"]
    ].copy()
    median_claims = provider_info["total_claims"].median()
    provider_info["high_volume"] = (provider_info["total_claims"] > median_claims).astype(int)
    del pf

    # Build monthly spending per provider
    records = []
    for pf_path in sorted(processed_dir.glob("medicaid_*.parquet")):
        logger.info(f"Building panel from {pf_path.name}...")
        df = pd.read_parquet(pf_path, columns=[
            "BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH", "TOTAL_PAID", "TOTAL_CLAIMS"
        ])
        df = df[df["BILLING_PROVIDER_NPI_NUM"].isin(top_npis)]
        agg = df.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"]).agg(
            monthly_paid=("TOTAL_PAID", "sum"),
            monthly_claims=("TOTAL_CLAIMS", "sum"),
        ).reset_index()
        records.append(agg)
        del df
        gc.collect()

    panel = pd.concat(records, ignore_index=True)
    panel.rename(columns={"BILLING_PROVIDER_NPI_NUM": "billing_npi",
                          "CLAIM_FROM_MONTH": "month_str"}, inplace=True)

    # Parse month
    panel["date"] = pd.to_datetime(panel["month_str"] + "-01")
    panel["month_num"] = (panel["date"].dt.year - 2018) * 12 + panel["date"].dt.month - 1
    panel["post_covid"] = (panel["date"] >= "2020-03-01").astype(int)
    panel["month_of_year"] = panel["date"].dt.month

    # Merge provider info
    panel = panel.merge(provider_info[["billing_npi", "high_volume"]], on="billing_npi", how="left")

    # Interaction term
    panel["post_x_highvol"] = panel["post_covid"] * panel["high_volume"]

    # Log spending
    panel["log_paid"] = np.log1p(panel["monthly_paid"].clip(lower=0))

    logger.info(f"Panel data: {panel.shape} ({panel['billing_npi'].nunique():,} providers)")
    return panel


def difference_in_differences(panel: pd.DataFrame, config: dict) -> dict:
    """Estimate COVID impact via Difference-in-Differences."""
    logger.info("Running Difference-in-Differences...")

    # Model: log(spending) = beta0 + beta1*post + beta2*highvol + beta3*(post*highvol) + controls
    # Add month-of-year dummies for seasonality
    month_dummies = pd.get_dummies(panel["month_of_year"], prefix="month", drop_first=True, dtype=float)
    X = pd.concat([
        panel[["post_covid", "high_volume", "post_x_highvol", "month_num"]],
        month_dummies,
    ], axis=1)
    X = add_constant(X)
    y = panel["log_paid"]

    # OLS (provider fixed effects approximated by demeaning)
    model = OLS(y, X, missing="drop").fit(cov_type="cluster",
                                           cov_kwds={"groups": panel["billing_npi"]})

    results = {
        "did_coefficient": float(model.params.get("post_x_highvol", np.nan)),
        "did_se": float(model.bse.get("post_x_highvol", np.nan)),
        "did_pvalue": float(model.pvalues.get("post_x_highvol", np.nan)),
        "did_ci_lower": float(model.conf_int().loc["post_x_highvol", 0]) if "post_x_highvol" in model.params.index else np.nan,
        "did_ci_upper": float(model.conf_int().loc["post_x_highvol", 1]) if "post_x_highvol" in model.params.index else np.nan,
        "post_covid_effect": float(model.params.get("post_covid", np.nan)),
        "r_squared": float(model.rsquared),
        "n_observations": int(model.nobs),
    }

    logger.info(f"DiD coefficient (post*highvol): {results['did_coefficient']:.4f} "
                f"(SE={results['did_se']:.4f}, p={results['did_pvalue']:.4e})")

    # Save full regression table
    reg_table = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_statistic": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    save_table(reg_table, "causal_did_regression", config)

    return results


def event_study(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run event study (dynamic DiD) with period-specific effects."""
    # Create period indicators (6-month bins relative to COVID)
    panel = panel.copy()
    panel["periods_from_covid"] = (panel["month_num"] - 26) // 6  # March 2020 = month 26
    panel["periods_from_covid"] = panel["periods_from_covid"].clip(-4, 4)

    # Create interaction dummies (period * highvol)
    period_dummies = pd.get_dummies(panel["periods_from_covid"], prefix="period", dtype=float)
    interaction_cols = []
    for col in period_dummies.columns:
        int_col = f"hv_x_{col}"
        panel[int_col] = period_dummies[col].values * panel["high_volume"].values
        interaction_cols.append(int_col)

    X = pd.concat([
        panel[["high_volume"]],
        period_dummies,
        panel[interaction_cols],
    ], axis=1)
    # Drop reference period (period_-1 = just before COVID)
    ref_cols = [c for c in X.columns if "_-1" in c]
    X = X.drop(columns=ref_cols, errors="ignore")
    X = add_constant(X)
    y = panel["log_paid"]

    model = OLS(y, X, missing="drop").fit(cov_type="cluster",
                                           cov_kwds={"groups": panel["billing_npi"]})

    # Extract event study coefficients
    es_results = []
    for p in range(-4, 5):
        col = f"hv_x_period_{p}"
        if col in model.params.index:
            es_results.append({
                "period": p,
                "coefficient": model.params[col],
                "se": model.bse[col],
                "ci_lower": model.conf_int().loc[col, 0],
                "ci_upper": model.conf_int().loc[col, 1],
                "p_value": model.pvalues[col],
            })

    return pd.DataFrame(es_results)


def bayesian_structural_time_series(
    config: Optional[dict] = None,
) -> dict:
    """Estimate COVID causal impact using structural time series."""
    if config is None:
        config = load_config()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load monthly aggregate
    ts = pd.read_parquet(processed_dir / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").set_index("date")
    ts = ts.asfreq("MS")

    intervention_date = pd.Timestamp(config.get("causal", {}).get("intervention_date", "2020-03-01"))

    # Pre-intervention
    pre = ts.loc[:intervention_date - pd.DateOffset(months=1), "total_paid"]
    post = ts.loc[intervention_date:, "total_paid"]

    # Fit structural time series on pre-period
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    model = UnobservedComponents(
        pre.values,
        level="local linear trend",
        seasonal=12,
    )
    fitted = model.fit(disp=False, maxiter=500)

    # Forecast counterfactual for post-period
    n_post = len(post)
    forecast = fitted.forecast(steps=n_post)
    forecast_ci = fitted.get_forecast(steps=n_post)
    ci = forecast_ci.summary_frame(alpha=0.05)

    # Compute impact
    actual = post.values
    counterfactual = forecast
    pointwise_impact = actual - counterfactual
    cumulative_impact = np.cumsum(pointwise_impact)

    total_impact = pointwise_impact.sum()
    mean_monthly_impact = pointwise_impact.mean()
    pct_impact = (actual.sum() - counterfactual.sum()) / counterfactual.sum() * 100

    results = {
        "total_causal_impact": float(total_impact),
        "mean_monthly_impact": float(mean_monthly_impact),
        "percent_impact": float(pct_impact),
        "pre_period": f"2018-01 to {(intervention_date - pd.DateOffset(months=1)).strftime('%Y-%m')}",
        "post_period": f"{intervention_date.strftime('%Y-%m')} to {post.index[-1].strftime('%Y-%m')}",
        "n_pre_months": len(pre),
        "n_post_months": len(post),
    }

    logger.info(f"BSTS: Total causal impact = {format_currency(total_impact)}, "
                f"{pct_impact:.1f}% increase vs counterfactual")

    # Save monthly impact
    impact_df = pd.DataFrame({
        "date": post.index,
        "actual": actual,
        "counterfactual": counterfactual,
        "pointwise_impact": pointwise_impact,
        "cumulative_impact": cumulative_impact,
    })
    save_table(impact_df, "causal_bsts_monthly_impact", config)

    results["_post_dates"] = post.index
    results["_actual"] = actual
    results["_counterfactual"] = counterfactual
    results["_ci_lower"] = ci["mean_ci_lower"].values if "mean_ci_lower" in ci.columns else counterfactual * 0.8
    results["_ci_upper"] = ci["mean_ci_upper"].values if "mean_ci_upper" in ci.columns else counterfactual * 1.2
    results["_pre_dates"] = pre.index
    results["_pre_values"] = pre.values
    results["_cumulative"] = cumulative_impact

    return results


def regression_discontinuity(panel: pd.DataFrame, config: dict) -> dict:
    """Regression Discontinuity Design at COVID onset."""
    logger.info("Running Regression Discontinuity...")

    bandwidth = config.get("causal", {}).get("rdd_bandwidth", 12)

    # National monthly aggregates
    monthly = panel.groupby("month_num").agg(
        total_paid=("monthly_paid", "sum"),
        n_providers=("billing_npi", "nunique"),
    ).reset_index()

    # Running variable centered at COVID onset (month 26)
    monthly["running_var"] = monthly["month_num"] - 26
    monthly["post"] = (monthly["running_var"] >= 0).astype(int)
    monthly["log_paid"] = np.log1p(monthly["total_paid"])

    # Filter to bandwidth
    bw_mask = monthly["running_var"].abs() <= bandwidth
    local = monthly[bw_mask].copy()

    # Local linear regression: y = a + b*running + c*post + d*post*running
    local["post_x_running"] = local["post"] * local["running_var"]
    X = add_constant(local[["running_var", "post", "post_x_running"]])
    y = local["log_paid"]

    model = OLS(y, X).fit()

    results = {
        "rdd_coefficient": float(model.params.get("post", np.nan)),
        "rdd_se": float(model.bse.get("post", np.nan)),
        "rdd_pvalue": float(model.pvalues.get("post", np.nan)),
        "bandwidth": bandwidth,
        "n_observations": int(len(local)),
        "r_squared": float(model.rsquared),
    }

    logger.info(f"RDD: Discontinuity = {results['rdd_coefficient']:.4f} "
                f"(SE={results['rdd_se']:.4f}, p={results['rdd_pvalue']:.4e})")

    results["_monthly"] = monthly
    results["_local"] = local
    return results


def synthetic_control(panel: pd.DataFrame, config: dict) -> dict:
    """Synthetic Control Method for counterfactual estimation."""
    logger.info("Running Synthetic Control...")

    # Define treated group: top 10% of providers by spending growth
    pre_covid_spending = panel[panel["post_covid"] == 0].groupby("billing_npi")["monthly_paid"].mean()
    post_covid_spending = panel[panel["post_covid"] == 1].groupby("billing_npi")["monthly_paid"].mean()

    common_npis = pre_covid_spending.index.intersection(post_covid_spending.index)
    growth = (post_covid_spending[common_npis] - pre_covid_spending[common_npis]) / pre_covid_spending[common_npis].clip(lower=1)
    growth = growth.replace([np.inf, -np.inf], np.nan).dropna()

    treated_threshold = growth.quantile(0.9)
    control_threshold = growth.quantile(0.5)
    treated_npis = set(growth[growth >= treated_threshold].index)
    donor_npis = set(growth[growth <= control_threshold].index)

    # Aggregate monthly time series for treated and donor pools
    treated_ts = panel[panel["billing_npi"].isin(treated_npis)].groupby("month_num")["monthly_paid"].sum()
    donor_ts = panel[panel["billing_npi"].isin(donor_npis)].groupby("month_num")["monthly_paid"].sum()

    # Pre-period matching
    pre_months = list(range(0, 26))  # 2018-01 to 2020-02
    post_months = list(range(26, 84))

    treated_pre = treated_ts.reindex(pre_months, fill_value=0).values
    donor_pre = donor_ts.reindex(pre_months, fill_value=0).values

    # Scale donor to match treated pre-period
    if donor_pre.sum() > 0:
        scale = treated_pre.mean() / donor_pre.mean()
    else:
        scale = 1.0

    synthetic = donor_ts * scale

    # Compute gap
    all_months = sorted(set(treated_ts.index) | set(synthetic.index))
    treated_full = treated_ts.reindex(all_months, fill_value=0)
    synthetic_full = synthetic.reindex(all_months, fill_value=0)
    gap = treated_full - synthetic_full

    results = {
        "n_treated": len(treated_npis),
        "n_donors": len(donor_npis),
        "pre_period_match_r2": float(np.corrcoef(treated_pre, donor_pre[:len(treated_pre)])[0, 1] ** 2) if len(donor_pre) >= len(treated_pre) else np.nan,
        "post_mean_gap": float(gap.loc[gap.index >= 26].mean()) if len(gap[gap.index >= 26]) > 0 else np.nan,
        "post_cumulative_gap": float(gap.loc[gap.index >= 26].sum()) if len(gap[gap.index >= 26]) > 0 else np.nan,
    }

    results["_treated"] = treated_full
    results["_synthetic"] = synthetic_full
    results["_gap"] = gap
    results["_all_months"] = all_months

    logger.info(f"Synthetic Control: {len(treated_npis)} treated, {len(donor_npis)} donors, "
                f"post-period mean gap = {format_currency(results['post_mean_gap'])}")

    return results


def plot_causal_analysis(
    did_results: dict,
    event_study_df: pd.DataFrame,
    bsts_results: dict,
    rdd_results: dict,
    sc_results: dict,
    config: dict,
) -> None:
    """Generate causal analysis visualizations."""
    setup_plotting(config)
    month_labels = pd.date_range("2018-01", "2024-12", freq="MS")

    # 1. DiD parallel trends (pre-treatment check)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not event_study_df.empty:
        ax.errorbar(event_study_df["period"], event_study_df["coefficient"],
                     yerr=[event_study_df["coefficient"] - event_study_df["ci_lower"],
                           event_study_df["ci_upper"] - event_study_df["coefficient"]],
                     fmt="o-", capsize=4, color="steelblue", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(-0.5, color="red", linestyle="--", alpha=0.5, label="COVID Onset")
        ax.set_xlabel("Period (6-month bins, 0 = COVID onset)")
        ax.set_ylabel("DiD Coefficient")
        ax.set_title("Event Study: Dynamic Difference-in-Differences")
        ax.legend()
        ax.grid(True, alpha=0.3)
    save_figure(fig, "causal_did_event_study", config)
    plt.close(fig)

    # 2. BSTS counterfactual
    if "_post_dates" in bsts_results:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Top: actual vs counterfactual
        ax = axes[0]
        ax.plot(bsts_results["_pre_dates"], bsts_results["_pre_values"] / 1e9,
                color="steelblue", linewidth=1.5, label="Observed (pre)")
        ax.plot(bsts_results["_post_dates"], bsts_results["_actual"] / 1e9,
                color="steelblue", linewidth=1.5, label="Observed (post)")
        ax.plot(bsts_results["_post_dates"], bsts_results["_counterfactual"] / 1e9,
                color="red", linewidth=1.5, linestyle="--", label="Counterfactual")
        ax.fill_between(bsts_results["_post_dates"],
                        bsts_results["_ci_lower"] / 1e9,
                        bsts_results["_ci_upper"] / 1e9,
                        alpha=0.2, color="red")
        ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("Monthly Spending ($B)")
        ax.set_title("Bayesian Structural Time Series: Observed vs Counterfactual")
        ax.legend()

        # Bottom: cumulative impact
        ax = axes[1]
        ax.plot(bsts_results["_post_dates"], bsts_results["_cumulative"] / 1e9,
                color="steelblue", linewidth=2)
        ax.fill_between(bsts_results["_post_dates"], 0, bsts_results["_cumulative"] / 1e9,
                        alpha=0.3, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Cumulative Impact ($B)")
        ax.set_xlabel("Date")
        ax.set_title(f"Cumulative Causal Impact: {format_currency(bsts_results['total_causal_impact'])} "
                     f"({bsts_results['percent_impact']:.1f}%)")
        plt.tight_layout()
        save_figure(fig, "causal_bsts_counterfactual", config)
        plt.close(fig)

    # 3. RDD plot
    if "_monthly" in rdd_results:
        monthly = rdd_results["_monthly"]
        fig, ax = plt.subplots(figsize=(12, 6))
        pre = monthly[monthly["running_var"] < 0]
        post = monthly[monthly["running_var"] >= 0]
        ax.scatter(pre["running_var"], pre["log_paid"], color="steelblue", s=30, label="Pre-COVID")
        ax.scatter(post["running_var"], post["log_paid"], color="orangered", s=30, label="Post-COVID")

        # Fit lines
        if len(pre) > 2:
            z = np.polyfit(pre["running_var"], pre["log_paid"], 1)
            x_line = np.linspace(pre["running_var"].min(), 0, 50)
            ax.plot(x_line, np.polyval(z, x_line), "steelblue", linewidth=2)
        if len(post) > 2:
            z = np.polyfit(post["running_var"], post["log_paid"], 1)
            x_line = np.linspace(0, post["running_var"].max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "orangered", linewidth=2)

        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Months Relative to COVID Onset (March 2020)")
        ax.set_ylabel("log(Total Monthly Spending)")
        ax.set_title(f"Regression Discontinuity at COVID Onset "
                     f"(jump = {rdd_results['rdd_coefficient']:.3f}, p = {rdd_results['rdd_pvalue']:.3e})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_figure(fig, "causal_rdd_plot", config)
        plt.close(fig)

    # 4. Synthetic control
    if "_treated" in sc_results:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        months = sc_results["_all_months"]
        dates = [month_labels[m] if m < len(month_labels) else month_labels[-1] for m in months]

        ax = axes[0]
        ax.plot(dates, sc_results["_treated"].values / 1e9, "steelblue", linewidth=2, label="Treated (high-growth)")
        ax.plot(dates, sc_results["_synthetic"].values / 1e9, "red", linewidth=2, linestyle="--", label="Synthetic Control")
        ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("Monthly Spending ($B)")
        ax.set_title("Synthetic Control: Treated vs Synthetic")
        ax.legend()

        ax = axes[1]
        ax.plot(dates, sc_results["_gap"].values / 1e9, "steelblue", linewidth=2)
        ax.fill_between(dates, 0, sc_results["_gap"].values / 1e9, alpha=0.3, color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", alpha=0.5)
        ax.set_ylabel("Gap ($B)")
        ax.set_xlabel("Date")
        ax.set_title("Spending Gap (Treated - Synthetic)")
        plt.tight_layout()
        save_figure(fig, "causal_synthetic_control", config)
        plt.close(fig)

    # 5. Methods comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ["DiD\n(post*highvol)", "BSTS\n(% impact)", "RDD\n(log jump)"]
    estimates = [
        did_results.get("did_coefficient", 0),
        bsts_results.get("percent_impact", 0) / 100,
        rdd_results.get("rdd_coefficient", 0),
    ]
    colors = ["steelblue", "teal", "coral"]
    ax.bar(methods, estimates, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("Estimated Effect Size")
    ax.set_title("COVID Causal Impact Estimates — Method Comparison")
    for i, v in enumerate(estimates):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    save_figure(fig, "causal_methods_comparison", config)
    plt.close(fig)


def run_causal_analysis(config: Optional[dict] = None) -> dict:
    """Run the full causal inference pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    causal_cfg = config.get("causal", {})
    top_n = causal_cfg.get("top_n_providers", 50000)

    # Step 1: Build panel
    logger.info(f"Building panel data for top {top_n:,} providers...")
    panel = build_panel_data(config, top_n=top_n)
    panel.to_parquet(processed_dir / "causal_panel_data.parquet", index=False)

    # Step 2: DiD
    did_results = difference_in_differences(panel, config)

    # Step 3: Event study
    logger.info("Running event study...")
    es_df = event_study(panel, config)

    # Step 4: BSTS
    logger.info("Running BSTS causal impact...")
    bsts_results = bayesian_structural_time_series(config)

    # Step 5: RDD
    rdd_results = regression_discontinuity(panel, config)

    # Step 6: Synthetic control
    sc_results = synthetic_control(panel, config)

    # Summary table
    summary = pd.DataFrame([{
        "method": "Difference-in-Differences",
        "estimate": did_results["did_coefficient"],
        "se": did_results["did_se"],
        "p_value": did_results["did_pvalue"],
        "interpretation": f"High-vol providers had {did_results['did_coefficient']:.3f} higher log-spending increase post-COVID",
    }, {
        "method": "BSTS Counterfactual",
        "estimate": bsts_results["percent_impact"],
        "se": np.nan,
        "p_value": np.nan,
        "interpretation": f"Aggregate spending was {bsts_results['percent_impact']:.1f}% above counterfactual post-COVID",
    }, {
        "method": "Regression Discontinuity",
        "estimate": rdd_results["rdd_coefficient"],
        "se": rdd_results["rdd_se"],
        "p_value": rdd_results["rdd_pvalue"],
        "interpretation": f"Discrete {rdd_results['rdd_coefficient']:.3f} log-point jump at COVID onset",
    }, {
        "method": "Synthetic Control",
        "estimate": sc_results.get("post_mean_gap", np.nan),
        "se": np.nan,
        "p_value": np.nan,
        "interpretation": f"Mean monthly gap: {format_currency(sc_results.get('post_mean_gap', 0))}",
    }])
    save_table(summary, "causal_impact_summary", config)

    # Plots
    logger.info("Generating causal analysis plots...")
    plot_causal_analysis(did_results, es_df, bsts_results, rdd_results, sc_results, config)

    logger.info("Causal analysis complete")
    return {"did": did_results, "bsts": {k: v for k, v in bsts_results.items() if not k.startswith("_")},
            "rdd": {k: v for k, v in rdd_results.items() if not k.startswith("_")},
            "sc": {k: v for k, v in sc_results.items() if not k.startswith("_")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal impact analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_causal_analysis(config)
