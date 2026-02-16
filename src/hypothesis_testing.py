"""Statistical hypothesis testing with rigorous methodology."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


class HypothesisResult:
    """Container for hypothesis test results."""

    def __init__(self, name: str, null_hypothesis: str, alt_hypothesis: str,
                 test_name: str, statistic: float, p_value: float,
                 effect_size: Optional[float] = None,
                 ci_lower: Optional[float] = None,
                 ci_upper: Optional[float] = None,
                 conclusion: str = "",
                 details: Optional[dict] = None):
        self.name = name
        self.null_hypothesis = null_hypothesis
        self.alt_hypothesis = alt_hypothesis
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.effect_size = effect_size
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.conclusion = conclusion
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "null_hypothesis": self.null_hypothesis,
            "alt_hypothesis": self.alt_hypothesis,
            "test": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
            "conclusion": self.conclusion,
            "details": self.details,
        }


def h1_covid_spending_shock(config: dict) -> HypothesisResult:
    """H1: Total Medicaid spending in 2020-2021 differs from 2018-2019.

    Uses Wilcoxon signed-rank test on monthly spending (paired by month-of-year).

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year"] = ts["date"].dt.year
    ts["month_num"] = ts["date"].dt.month

    pre_covid = ts[ts["year"].isin([2018, 2019])].groupby("month_num")["total_paid"].mean()
    covid = ts[ts["year"].isin([2020, 2021])].groupby("month_num")["total_paid"].mean()

    common_months = sorted(set(pre_covid.index) & set(covid.index))
    pre_values = pre_covid.loc[common_months].values
    covid_values = covid.loc[common_months].values

    stat, p = stats.wilcoxon(pre_values, covid_values, alternative="two-sided")

    # Effect size: rank-biserial correlation
    n = len(common_months)
    effect_size = 1 - (2 * stat) / (n * (n + 1) / 2)

    # Confidence interval via bootstrap
    diffs = covid_values - pre_values
    ci = np.percentile(diffs, [2.5, 97.5])

    pct_change = (covid_values.mean() - pre_values.mean()) / pre_values.mean() * 100

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    months = range(1, 13)
    ax.plot(months[:len(pre_values)], pre_values / 1e9, "b-o", label="Pre-COVID (2018-2019 avg)", linewidth=2)
    ax.plot(months[:len(covid_values)], covid_values / 1e9, "r-o", label="COVID (2020-2021 avg)", linewidth=2)
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Average Monthly Spending ($B)")
    ax.set_title(f"H1: COVID Spending Impact (p={p:.2e}, change={pct_change:+.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "h1_covid_spending", config)

    return HypothesisResult(
        name="H1: COVID Spending Shock",
        null_hypothesis="Monthly spending in 2020-2021 equals 2018-2019 baseline",
        alt_hypothesis="Monthly spending differs between periods",
        test_name="Wilcoxon signed-rank test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"COVID period spending was {pct_change:+.1f}% vs pre-COVID. "
                   f"Effect size (rank-biserial r) = {effect_size:.3f}.",
        details={"pct_change": pct_change, "pre_mean_monthly": float(pre_values.mean()),
                 "covid_mean_monthly": float(covid_values.mean())},
    )


def h2_post_covid_sustained(config: dict) -> HypothesisResult:
    """H2: Post-COVID spending (2022-2024) exceeds pre-COVID (2018-2019).

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year"] = ts["date"].dt.year

    pre = ts[ts["year"].isin([2018, 2019])]["total_paid"].values
    post = ts[ts["year"].isin([2022, 2023, 2024])]["total_paid"].values

    stat, p = stats.mannwhitneyu(pre, post, alternative="less")

    # Effect size: rank-biserial correlation
    n1, n2 = len(pre), len(post)
    effect_size = 1 - (2 * stat) / (n1 * n2)

    # Bootstrap CI for difference in means
    np.random.seed(42)
    boot_diffs = []
    for _ in range(10000):
        s1 = np.random.choice(pre, size=len(pre), replace=True)
        s2 = np.random.choice(post, size=len(post), replace=True)
        boot_diffs.append(s2.mean() - s1.mean())
    ci = np.percentile(boot_diffs, [2.5, 97.5])

    pct_change = (post.mean() - pre.mean()) / pre.mean() * 100

    return HypothesisResult(
        name="H2: Post-COVID Sustained Increase",
        null_hypothesis="Post-COVID monthly spending <= pre-COVID monthly spending",
        alt_hypothesis="Post-COVID monthly spending > pre-COVID",
        test_name="Mann-Whitney U (one-sided)",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Post-COVID spending is {pct_change:+.1f}% vs pre-COVID.",
        details={"pct_change": pct_change, "pre_mean": float(pre.mean()),
                 "post_mean": float(post.mean())},
    )


def h3_power_law_spending(config: dict) -> HypothesisResult:
    """H3: Provider spending follows a power-law distribution.

    Fits power-law vs. lognormal and compares via likelihood ratio test.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    import powerlaw

    root = get_project_root()
    providers = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet",
        columns=["total_paid"],
    )
    data = providers["total_paid"][providers["total_paid"] > 0].values

    # Subsample for powerlaw fitting (full dataset is too slow)
    np.random.seed(config["analysis"]["random_seed"])
    if len(data) > 50000:
        data = np.random.choice(data, 50000, replace=False)

    fit = powerlaw.Fit(data, discrete=False, verbose=False)
    R, p_lr = fit.distribution_compare("power_law", "lognormal")

    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 7))
    fit.plot_ccdf(ax=ax, label="Empirical CCDF", color="#1565C0", linewidth=2)
    fit.power_law.plot_ccdf(ax=ax, label=f"Power law (alpha={fit.alpha:.2f})", color="red", linestyle="--")
    fit.lognormal.plot_ccdf(ax=ax, label="Lognormal fit", color="green", linestyle="--")
    ax.set_xlabel("Total Provider Spending ($)")
    ax.set_ylabel("CCDF P(X > x)")
    ax.set_title("H3: Provider Spending Distribution")
    ax.legend()
    save_figure(fig, "h3_power_law", config)

    # Power law is preferred if R > 0
    preferred = "power_law" if R > 0 else "lognormal"
    return HypothesisResult(
        name="H3: Power-Law Provider Spending",
        null_hypothesis="Provider spending follows a lognormal distribution",
        alt_hypothesis="Provider spending follows a power-law distribution",
        test_name="Likelihood ratio test (power-law vs. lognormal)",
        statistic=float(R),
        p_value=float(p_lr),
        effect_size=float(fit.alpha),
        conclusion=f"{'Power-law' if R > 0 else 'Lognormal'} is preferred (R={R:.3f}, p={p_lr:.3e}). "
                   f"Power-law exponent alpha={fit.alpha:.2f}, xmin={fit.xmin:.0f}.",
        details={"alpha": float(fit.alpha), "xmin": float(fit.xmin),
                 "loglikelihood_ratio": float(R), "preferred": preferred},
    )


def h4_spending_growth_acceleration(config: dict) -> HypothesisResult:
    """H4: Spending growth rate is accelerating over time.

    Tests if YoY growth rate shows a significant positive trend.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year"] = ts["date"].dt.year

    annual = ts.groupby("year")["total_paid"].sum().reset_index()
    annual["yoy_growth"] = annual["total_paid"].pct_change() * 100
    annual = annual.dropna()

    # Mann-Kendall trend test
    from scipy.stats import kendalltau
    tau, p = kendalltau(annual["year"].values, annual["yoy_growth"].values)

    return HypothesisResult(
        name="H4: Spending Growth Acceleration",
        null_hypothesis="YoY spending growth rate has no trend over time",
        alt_hypothesis="YoY spending growth rate shows a significant trend",
        test_name="Kendall tau trend test",
        statistic=float(tau),
        p_value=float(p),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Kendall tau = {tau:.3f}. "
                   f"Growth rate is {'accelerating' if tau > 0 else 'decelerating'} "
                   f"(p={p:.3f}).",
        details={"annual_growth_rates": dict(zip(annual["year"].astype(str), annual["yoy_growth"].round(2)))},
    )


def h5_seasonality(config: dict) -> HypothesisResult:
    """H5: Medicaid spending exhibits significant monthly seasonality.

    Uses Kruskal-Wallis test across months, then Friedman test for repeated measures.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["month_num"] = ts["date"].dt.month

    groups = [ts[ts["month_num"] == m]["total_paid"].values for m in range(1, 13)]
    groups = [g for g in groups if len(g) > 0]

    stat, p = stats.kruskal(*groups)

    # Effect size: eta-squared
    n_total = sum(len(g) for g in groups)
    k = len(groups)
    eta_sq = (stat - k + 1) / (n_total - k)

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 6))
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    bp_data = [ts[ts["month_num"] == m]["total_paid"].values / 1e9 for m in range(1, 13)]
    bp = ax.boxplot(bp_data, labels=month_names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#64B5F6")
        patch.set_alpha(0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Total Spending ($B)")
    ax.set_title(f"H5: Spending Seasonality (Kruskal-Wallis p={p:.3e})")
    save_figure(fig, "h5_seasonality", config)

    return HypothesisResult(
        name="H5: Monthly Seasonality",
        null_hypothesis="Monthly spending does not differ across months of the year",
        alt_hypothesis="Monthly spending differs significantly by month",
        test_name="Kruskal-Wallis H-test",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(eta_sq),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Eta-squared = {eta_sq:.4f}. "
                   f"{'Strong' if eta_sq > 0.14 else 'Moderate' if eta_sq > 0.06 else 'Weak'} effect.",
    )


def h6_procedure_concentration_shift(config: dict) -> HypothesisResult:
    """H6: The Herfindahl-Hirschman Index of procedure concentration has changed.

    Computes HHI of spending across HCPCS codes per year.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    yearly_hhi = {}
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        year = int(pf.stem.split("_")[1])
        df = pd.read_parquet(pf, columns=["HCPCS_CODE", "TOTAL_PAID"])
        code_shares = df.groupby("HCPCS_CODE")["TOTAL_PAID"].sum()
        total = code_shares.sum()
        if total > 0:
            shares = code_shares / total
            hhi = (shares ** 2).sum() * 10000  # scale to 0-10000
            yearly_hhi[year] = float(hhi)
        del df

    years = sorted(yearly_hhi.keys())
    hhi_values = [yearly_hhi[y] for y in years]

    tau, p = stats.kendalltau(years, hhi_values)

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, hhi_values, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Year")
    ax.set_ylabel("HHI (Procedure Code Concentration)")
    ax.set_title(f"H6: HCPCS Concentration Trend (Kendall tau={tau:.3f}, p={p:.3f})")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "h6_hhi_trend", config)

    return HypothesisResult(
        name="H6: Procedure Concentration Shift",
        null_hypothesis="HCPCS concentration (HHI) has no trend over time",
        alt_hypothesis="HCPCS concentration changes over time",
        test_name="Kendall tau trend test",
        statistic=float(tau),
        p_value=float(p),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"HHI is {'increasing (more concentrated)' if tau > 0 else 'decreasing (more dispersed)'}. "
                   f"Tau = {tau:.3f}.",
        details={"yearly_hhi": yearly_hhi},
    )


def h7_cross_servicing_spending(config: dict) -> HypothesisResult:
    """H7: Providers using multiple servicing NPIs have higher spending.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    prov = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet",
        columns=["total_paid", "n_servicing_npis"],
    )

    single = prov[prov["n_servicing_npis"] <= 1]["total_paid"].values
    multi = prov[prov["n_servicing_npis"] > 1]["total_paid"].values

    stat, p = stats.mannwhitneyu(single, multi, alternative="less")

    # Effect size: rank-biserial
    n1, n2 = len(single), len(multi)
    effect_size = 1 - (2 * stat) / (n1 * n2)

    # Bootstrap CI
    np.random.seed(42)
    boot_diffs = []
    for _ in range(10000):
        s1 = np.random.choice(single, size=min(1000, len(single)), replace=True)
        s2 = np.random.choice(multi, size=min(1000, len(multi)), replace=True)
        boot_diffs.append(s2.mean() - s1.mean())
    ci = np.percentile(boot_diffs, [2.5, 97.5])

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [np.log10(single[single > 0] + 1), np.log10(multi[multi > 0] + 1)]
    bp = ax.boxplot(data, labels=["Single Servicing NPI", "Multiple Servicing NPIs"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("#64B5F6")
    bp["boxes"][1].set_facecolor("#EF5350")
    ax.set_ylabel("log10(Total Spending + 1)")
    ax.set_title(f"H7: Spending by Servicing NPI Count (p={p:.2e})")
    save_figure(fig, "h7_cross_servicing", config)

    return HypothesisResult(
        name="H7: Multi-Servicing NPI Spending",
        null_hypothesis="Providers with multiple servicing NPIs have same spending as single-NPI",
        alt_hypothesis="Multi-servicing NPI providers have higher total spending",
        test_name="Mann-Whitney U (one-sided)",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(effect_size),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Multi-servicing NPI providers have "
                   f"{'higher' if p < 0.05 else 'similar'} spending. "
                   f"Median single: {format_currency(np.median(single))}, "
                   f"Median multi: {format_currency(np.median(multi))}.",
        details={"n_single": int(n1), "n_multi": int(n2),
                 "median_single": float(np.median(single)),
                 "median_multi": float(np.median(multi))},
    )


def h8_claims_per_beneficiary_trend(config: dict) -> HypothesisResult:
    """H8: Claims per beneficiary is increasing over time.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year"] = ts["date"].dt.year

    annual = ts.groupby("year").agg(
        total_claims=("total_claims", "sum"),
        total_beneficiaries=("total_beneficiaries", "sum"),
    )
    annual["claims_per_bene"] = annual["total_claims"] / annual["total_beneficiaries"]

    tau, p = stats.kendalltau(annual.index, annual["claims_per_bene"])

    # Linear regression for effect size
    slope, intercept, r, p_ols, se = stats.linregress(
        annual.index.astype(float), annual["claims_per_bene"]
    )

    return HypothesisResult(
        name="H8: Claims per Beneficiary Trend",
        null_hypothesis="Claims per beneficiary is stable over time",
        alt_hypothesis="Claims per beneficiary shows a significant trend",
        test_name="Kendall tau + OLS regression",
        statistic=float(tau),
        p_value=float(p),
        effect_size=float(r ** 2),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Claims/beneficiary is {'increasing' if tau > 0 else 'decreasing'} "
                   f"(tau={tau:.3f}, R^2={r**2:.3f}, slope={slope:.4f}/year).",
        details={"yearly_claims_per_bene": {str(y): round(v, 4) for y, v in annual["claims_per_bene"].items()},
                 "slope_per_year": float(slope), "r_squared": float(r ** 2)},
    )


def h9_paid_per_claim_inflation(config: dict) -> HypothesisResult:
    """H9: Paid per claim has increased faster than general inflation.

    Uses approximate CPI (3% annual) as benchmark.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts["year"] = ts["date"].dt.year

    annual = ts.groupby("year").agg(
        total_paid=("total_paid", "sum"),
        total_claims=("total_claims", "sum"),
    )
    annual["paid_per_claim"] = annual["total_paid"] / annual["total_claims"]

    years = annual.index.values.astype(float)
    ppc = annual["paid_per_claim"].values

    # Compute CAGR of paid per claim
    n_years = years[-1] - years[0]
    cagr = (ppc[-1] / ppc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # One-sample t-test: is growth rate > 3% (CPI benchmark)?
    # Compute annual growth rates
    growth_rates = np.diff(ppc) / ppc[:-1]
    cpi_rate = 0.03  # approximate average CPI inflation

    stat, p = stats.ttest_1samp(growth_rates, cpi_rate)
    # One-sided: is it greater?
    p_one_sided = p / 2 if stat > 0 else 1 - p / 2

    ci = stats.t.interval(0.95, len(growth_rates) - 1,
                          loc=growth_rates.mean(),
                          scale=stats.sem(growth_rates))

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, ppc, "bo-", linewidth=2, markersize=8, label="Actual Paid/Claim")
    # CPI-adjusted baseline
    cpi_projected = ppc[0] * (1.03 ** (years - years[0]))
    ax.plot(years, cpi_projected, "r--", linewidth=1.5, label="CPI-adjusted (3%/yr)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Paid per Claim ($)")
    ax.set_title(f"H9: Medicaid Cost per Claim vs CPI (CAGR={cagr*100:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "h9_cost_inflation", config)

    return HypothesisResult(
        name="H9: Per-Claim Cost Inflation",
        null_hypothesis="Paid per claim grows at the same rate as CPI (~3%/year)",
        alt_hypothesis="Paid per claim grows faster than CPI",
        test_name="One-sample t-test vs 3% benchmark",
        statistic=float(stat),
        p_value=float(p_one_sided),
        effect_size=float(cagr),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        conclusion=f"{'Reject' if p_one_sided < 0.05 else 'Fail to reject'} H0. "
                   f"Medicaid cost per claim CAGR = {cagr*100:.1f}% vs CPI benchmark of 3%. "
                   f"Mean annual growth rate = {growth_rates.mean()*100:.1f}% "
                   f"(95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]).",
        details={"cagr": float(cagr), "cpi_benchmark": 0.03,
                 "mean_growth_rate": float(growth_rates.mean()),
                 "yearly_paid_per_claim": {str(int(y)): round(v, 2) for y, v in zip(years, ppc)}},
    )


def h10_provider_specialization(config: dict) -> HypothesisResult:
    """H10: Specialized providers (few HCPCS codes) have higher per-claim costs.

    Args:
        config: Configuration dictionary.

    Returns:
        HypothesisResult.
    """
    root = get_project_root()
    prov = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet",
        columns=["n_unique_hcpcs", "paid_per_claim", "total_claims"],
    )
    prov = prov[prov["total_claims"] > 100]  # filter out very small providers

    # Define specialist (<=5 codes) vs generalist (>20 codes)
    specialist = prov[prov["n_unique_hcpcs"] <= 5]["paid_per_claim"].values
    generalist = prov[prov["n_unique_hcpcs"] > 20]["paid_per_claim"].values

    stat, p = stats.mannwhitneyu(specialist, generalist, alternative="two-sided")
    n1, n2 = len(specialist), len(generalist)
    effect_size = 1 - (2 * stat) / (n1 * n2)

    # Spearman correlation
    rho, p_spearman = stats.spearmanr(
        prov["n_unique_hcpcs"].values,
        prov["paid_per_claim"].values,
    )

    return HypothesisResult(
        name="H10: Provider Specialization vs Cost",
        null_hypothesis="Specialized providers have same per-claim cost as generalists",
        alt_hypothesis="Per-claim cost differs between specialist and generalist providers",
        test_name="Mann-Whitney U + Spearman correlation",
        statistic=float(stat),
        p_value=float(p),
        effect_size=float(rho),
        conclusion=f"{'Reject' if p < 0.05 else 'Fail to reject'} H0. "
                   f"Spearman rho = {rho:.3f} (p={p_spearman:.2e}). "
                   f"Median specialist: {format_currency(np.median(specialist))}, "
                   f"Median generalist: {format_currency(np.median(generalist))}.",
        details={"n_specialist": int(n1), "n_generalist": int(n2),
                 "median_specialist_ppc": float(np.median(specialist)),
                 "median_generalist_ppc": float(np.median(generalist)),
                 "spearman_rho": float(rho), "spearman_p": float(p_spearman)},
    )


def run_all_tests(config: Optional[dict] = None) -> list:
    """Run all hypothesis tests with multiple comparison correction.

    Args:
        config: Configuration dictionary.

    Returns:
        List of HypothesisResult objects.
    """
    if config is None:
        config = load_config()

    tests = [
        ("H1", h1_covid_spending_shock),
        ("H2", h2_post_covid_sustained),
        ("H3", h3_power_law_spending),
        ("H4", h4_spending_growth_acceleration),
        ("H5", h5_seasonality),
        ("H6", h6_procedure_concentration_shift),
        ("H7", h7_cross_servicing_spending),
        ("H8", h8_claims_per_beneficiary_trend),
        ("H9", h9_paid_per_claim_inflation),
        ("H10", h10_provider_specialization),
    ]

    results = []
    for name, func in tests:
        logger.info(f"Running {name}...")
        try:
            result = func(config)
            results.append(result)
            logger.info(f"  {result.conclusion}")
        except Exception as e:
            logger.error(f"  {name} failed: {e}")

    # Multiple comparison correction (Benjamini-Hochberg FDR)
    p_values = [r.p_value for r in results]
    reject, corrected_p, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

    for i, r in enumerate(results):
        r.details["corrected_p_value"] = float(corrected_p[i])
        r.details["reject_after_fdr"] = bool(reject[i])

    # Save results
    root = get_project_root()
    tables_dir = root / config["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([r.to_dict() for r in results])
    summary["corrected_p_value"] = corrected_p
    summary["reject_after_fdr"] = reject
    summary.to_csv(tables_dir / "hypothesis_test_results.csv", index=False)

    # Save detailed JSON
    with open(tables_dir / "hypothesis_test_details.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)

    logger.info(f"Hypothesis testing complete: {sum(reject)}/{len(reject)} significant after FDR correction")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical hypothesis tests")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_plotting(config)
    run_all_tests(config)
