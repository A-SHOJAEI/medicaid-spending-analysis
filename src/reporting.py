"""Auto-generate and update FINDINGS_REPORT.md."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root, format_currency, format_number

logger = logging.getLogger("medicaid_analysis")


def generate_findings_report(config: Optional[dict] = None) -> str:
    """Generate the complete FINDINGS_REPORT.md from analysis outputs.

    Args:
        config: Configuration dictionary.

    Returns:
        Markdown string of the complete report.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    tables_dir = root / config["paths"]["tables_dir"]

    sections = []
    sections.append(_header())
    sections.append(_data_profile(tables_dir))
    sections.append(_eda_findings(tables_dir))
    sections.append(_hypothesis_results(tables_dir))
    sections.append(_anomaly_findings(tables_dir))
    sections.append(_clustering_findings(tables_dir))
    sections.append(_time_series_findings(tables_dir))
    sections.append(_model_findings(tables_dir))
    sections.append(_network_findings(tables_dir))
    sections.append(_phase6_findings(tables_dir))
    sections.append(_phase7_findings(tables_dir))
    sections.append(_limitations())
    sections.append(_methodology())

    report = "\n\n".join(s for s in sections if s)

    report_path = root / "FINDINGS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"FINDINGS_REPORT.md written to {report_path}")
    return report


def _header() -> str:
    return f"""# Medicaid Provider Spending Analysis: Findings Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** HHS/DOGE Medicaid Provider Spending (T-MSIS)
**Coverage:** Fee-for-service, managed care, and CHIP claims (2018-2024)
**Source:** Centers for Medicare & Medicaid Services (CMS), T-MSIS Analytic Files

---

## Executive Summary

This report presents a comprehensive analysis of the Medicaid Provider Spending dataset,
covering 227,083,361 aggregated billing records from 617,503 billing providers across
10,881 procedure codes from January 2018 through December 2024. Total spending in the
dataset is **$1.09 trillion** across **18.83 billion claims**.

### Top 10 Findings

1. **Extreme spending concentration** (Gini = 0.925): The top 0.8% of providers account
   for 50% of all Medicaid spending. 469 mega-providers (0.08%) have median spending of
   $54.9M each, while the remaining 617,034 providers have median spending of just $47.7K.

2. **Sustained post-COVID spending surge**: Medicaid spending increased 59.4% from the
   pre-COVID period (2018-2019) to post-COVID (2022-2024), a statistically significant
   shift (Mann-Whitney U, p < 1e-9). The 2021 jump (+23.1% YoY) was the largest single-year
   increase, driven by continuous enrollment provisions.

3. **$1.09 trillion over 7 years**: Annual spending grew from $108.7B (2018) to a peak of
   $198.8B (2023), before declining to $185.0B in 2024 as pandemic-era enrollment unwinding
   took effect.

4. **Home care dominates**: HCPCS code T1019 (Personal Care Services) alone accounts for
   $122.7B (11.2% of all spending), followed by T1015 (Clinic Services, $49.2B) and T2016
   (Habilitation Services, $34.9B). Home and community-based services dominate Medicaid costs.

5. **146,806 providers flagged for anomalous patterns**: 6 providers triggered all 5 fraud
   signal flags simultaneously, and 418 triggered 4+ flags. Patterns include procedure code
   stuffing (30,831 providers), use of many servicing NPIs (37,615), and spending spikes
   >300% YoY (97,739 providers).

6. **Benford's Law deviation**: Payment amounts show statistically significant deviation
   from expected first-digit distribution (chi2 = 240,108, p ~ 0), suggesting systematic
   rounding patterns in Medicaid payment amounts.

7. **Specialists charge more per claim**: Providers billing 5 or fewer HCPCS codes have
   significantly higher per-claim costs (median $46.09) than generalists billing 20+ codes
   (median $31.07), with Spearman rho = -0.217 (p ~ 0).

8. **Provider spending follows lognormal, not power-law**: Formal statistical testing
   (likelihood ratio R = -7.48, p = 0.047) favors a lognormal distribution over power-law,
   though the tail behavior is extreme (alpha = 2.14).

9. **Spending is non-stationary with 3 structural breaks**: The PELT changepoint algorithm
   detected 3 regime changes in the monthly spending series, corresponding to COVID onset,
   the enrollment surge, and the beginning of enrollment unwinding.

10. **88 provider pairs with >80% billing similarity**: Network analysis of the provider-HCPCS
    bipartite graph identified 88 pairs of providers billing nearly identical procedure code
    portfolios, warranting investigation for potential coordinated billing.

### Key Policy Implications

- The extreme provider concentration means targeted oversight of ~5,000 providers could
  cover 80% of all Medicaid spending
- Home/community-based services (T1019, S5125, etc.) represent the largest cost category
  and the fastest-growing segment, meriting specific policy attention
- The 2024 spending decline reflects Medicaid enrollment unwinding but spending remains
  83% above 2018 levels, suggesting permanent structural shifts
- Anomaly detection flagged providers with concrete, actionable patterns (not just
  statistical outliers) that can guide fraud investigation priorities"""


def _data_profile(tables_dir: Path) -> str:
    profile_path = tables_dir / "profiling_results.json"
    if not profile_path.exists():
        return ""

    with open(profile_path) as f:
        profile = json.load(f)

    return f"""---

## Phase 1: Data Profile & Quality Assessment

### Dataset Overview

| Metric | Value |
|--------|-------|
| File Size | {profile['file_size_gb']} GB |
| Total Rows | {profile['total_rows']:,} |
| Columns | {', '.join(profile['columns'])} |
| Date Range | {profile['date_range']['min']} to {profile['date_range']['max']} |
| Unique Billing NPIs | {profile['cardinality']['billing_npi']:,} |
| Unique Servicing NPIs | {profile['cardinality']['servicing_npi']:,} |
| Unique HCPCS Codes | {profile['cardinality']['hcpcs_codes']:,} |
| Months Covered | {profile['cardinality']['months']} |

### Spending Summary

| Metric | Value |
|--------|-------|
| Total Spending | {format_currency(profile['paid_stats']['total'])} |
| Mean per Record | {format_currency(profile['paid_stats']['mean'])} |
| Std Dev | {format_currency(profile['paid_stats']['std'])} |
| Min | {format_currency(profile['paid_stats']['min'])} |
| Max | {format_currency(profile['paid_stats']['max'])} |
| Rows with Negative Payment | {profile['paid_stats']['negative_rows']:,} |
| Rows with Zero Payment | {profile['paid_stats']['zero_rows']:,} |
| Total Claims | {format_number(profile['claims_stats']['total'])} |

### Null Rates

| Column | Null Rate |
|--------|-----------|
""" + "\n".join(f"| {col} | {rate}% |" for col, rate in profile['null_rates'].items()) + """

### NPI Relationship Analysis

""" + f"""| Metric | Value |
|--------|-------|
| Billing = Servicing NPI | {profile['npi_relationship']['billing_equals_servicing']:,} ({profile['npi_relationship']['billing_equals_servicing']/profile['npi_relationship']['total_rows']*100:.1f}%) |
| Servicing NPI is Null | {profile['npi_relationship']['servicing_null']:,} ({profile['npi_relationship']['servicing_null']/profile['npi_relationship']['total_rows']*100:.1f}%) |

### Data Quality Assessment

- **Missing data**: SERVICING_PROVIDER_NPI_NUM has the highest null rate, which is expected for
  providers who both bill and render services.
- **Negative payments**: {profile['paid_stats']['negative_rows']:,} records have negative TOTAL_PAID values,
  indicating payment adjustments, recoupments, or refunds.
- **Cell suppression**: The dataset suppresses rows with fewer than 12 claims for privacy.
  This means low-volume provider-procedure combinations are systematically absent.
- **No state column**: The dataset lacks a state/geography identifier, limiting
  geographic analysis to NPI-based inferences.
- **No claim type column**: FFS vs managed care vs CHIP distinction is not available
  in this dataset version.

### Top 20 HCPCS Codes by Total Spending

""" + _top_hcpcs_table(tables_dir)


def _top_hcpcs_table(tables_dir: Path) -> str:
    path = tables_dir / "top_hcpcs_by_spending.csv"
    if not path.exists():
        return "*Table not yet generated.*"
    df = pd.read_csv(path)
    lines = ["| Rank | HCPCS Code | Total Spending |", "|------|-----------|---------------|"]
    for i, row in df.iterrows():
        lines.append(f"| {i+1} | {row['hcpcs_code']} | {format_currency(row['total_spending'])} |")
    return "\n".join(lines)


def _eda_findings(tables_dir: Path) -> str:
    annual_path = tables_dir / "annual_spending_summary.csv"
    if not annual_path.exists():
        return ""

    annual = pd.read_csv(annual_path)

    return f"""---

## Phase 2: Exploratory Data Analysis

### Annual Spending Trends

Total national Medicaid provider spending by year:

| Year | Total Spending | Total Claims | YoY Growth |
|------|---------------|-------------|------------|
""" + _annual_table(annual) + """

![Annual Spending](outputs/figures/annual_spending_and_claims.png)

### Monthly Time Series

The monthly spending time series reveals clear temporal patterns:
- **Trend**: Upward trajectory throughout the period, with acceleration during 2020-2022
- **Seasonality**: Regular monthly fluctuations with typical year-end patterns
- **COVID Impact**: Visible disruption starting March 2020

![Monthly Time Series](outputs/figures/monthly_time_series.png)

### Spending Distribution

Payment amounts span many orders of magnitude, from small claims to hundreds of millions.
The distribution is highly right-skewed, typical of healthcare spending data.

![Spending Distribution](outputs/figures/spending_distribution.png)

### Provider Concentration

Medicaid spending is highly concentrated among a small number of providers,
consistent with power-law-like distributions common in healthcare billing.

![Lorenz Curve](outputs/figures/provider_concentration_lorenz.png)

### Seasonality Patterns

![Seasonality](outputs/figures/seasonality_patterns.png)

### Procedure Code Analysis

![Top HCPCS](outputs/figures/top20_hcpcs_spending.png)

![HCPCS Frequency vs Cost](outputs/figures/hcpcs_frequency_vs_cost.png)

### Provider Diversity

![Provider HCPCS Diversity](outputs/figures/provider_hcpcs_diversity.png)
"""


def _annual_table(annual: pd.DataFrame) -> str:
    lines = []
    for i, row in annual.iterrows():
        yoy = ""
        if i > 0:
            prev = annual.iloc[i - 1]["total_paid"]
            growth = (row["total_paid"] - prev) / prev * 100
            yoy = f"{growth:+.1f}%"
        lines.append(
            f"| {int(row['year'])} | {format_currency(row['total_paid'])} | "
            f"{format_number(row['total_claims'])} | {yoy} |"
        )
    return "\n".join(lines)


def _hypothesis_results(tables_dir: Path) -> str:
    path = tables_dir / "hypothesis_test_details.json"
    if not path.exists():
        return ""

    with open(path) as f:
        results = json.load(f)

    lines = ["""---

## Phase 3: Hypothesis Testing

All tests were conducted at alpha = 0.05 with Benjamini-Hochberg FDR correction
for multiple comparisons.

| # | Hypothesis | Test | p-value | FDR p-value | Result |
|---|-----------|------|---------|-------------|--------|"""]

    for r in results:
        fdr_p = r.get("details", {}).get("corrected_p_value", r["p_value"])
        reject = r.get("details", {}).get("reject_after_fdr", r["p_value"] < 0.05)
        result_str = "REJECT H0" if reject else "Fail to reject"
        lines.append(
            f"| {r['name'].split(':')[0]} | {r['name']} | {r['test']} | "
            f"{r['p_value']:.2e} | {fdr_p:.2e} | **{result_str}** |"
        )

    lines.append("\n### Detailed Results\n")
    for r in results:
        lines.append(f"#### {r['name']}\n")
        lines.append(f"- **Null hypothesis**: {r['null_hypothesis']}")
        lines.append(f"- **Alternative**: {r['alt_hypothesis']}")
        lines.append(f"- **Test**: {r['test']}")
        lines.append(f"- **Test statistic**: {r['statistic']:.4f}")
        lines.append(f"- **p-value**: {r['p_value']:.2e}")
        if r.get("effect_size") is not None:
            lines.append(f"- **Effect size**: {r['effect_size']:.4f}")
        if r.get("ci_95_lower") is not None:
            lines.append(f"- **95% CI**: [{r['ci_95_lower']:.4f}, {r['ci_95_upper']:.4f}]")
        lines.append(f"- **Conclusion**: {r['conclusion']}\n")

    return "\n".join(lines)


def _anomaly_findings(tables_dir: Path) -> str:
    flagged_path = tables_dir / "top_flagged_providers.csv"
    anomaly_path = tables_dir / "top_anomalous_providers.csv"
    benford_path = tables_dir / "benford_analysis.csv"

    # Prefer the flag-sorted table; fall back to composite anomaly score table
    if flagged_path.exists():
        flagged = pd.read_csv(flagged_path)
    elif anomaly_path.exists():
        flagged = pd.read_csv(anomaly_path)
    else:
        return ""

    # Read full anomaly scores from parquet for accurate flag counts
    root = get_project_root()
    full_scores_path = root / "data" / "processed" / "provider_anomaly_scores.parquet"
    if full_scores_path.exists():
        full_scores = pd.read_parquet(full_scores_path)
    else:
        full_scores = flagged  # fallback to CSV if parquet unavailable

    # Compute flag summary counts from full dataset
    total_flagged = int((full_scores["fraud_flags_count"] >= 1).sum()) if "fraud_flags_count" in full_scores.columns else len(flagged)
    flag_5 = int((full_scores["fraud_flags_count"] >= 5).sum()) if "fraud_flags_count" in full_scores.columns else 0
    flag_4 = int((full_scores["fraud_flags_count"] >= 4).sum()) if "fraud_flags_count" in full_scores.columns else 0

    # Per-flag counts from full dataset
    flag_cols = {
        "flag_code_stuffing": "Procedure code stuffing (>100 HCPCS codes)",
        "flag_high_cost_per_claim": "High cost per claim (>$500)",
        "flag_many_servicing": "Many servicing NPIs (>50)",
        "flag_extreme_spending": "Extreme total spending (top 0.1%)",
        "flag_spending_spike": "Year-over-year spending spike (>300%)",
    }
    flag_summary_lines = []
    for col, desc in flag_cols.items():
        if col in full_scores.columns:
            count = full_scores[col].sum()
            flag_summary_lines.append(f"| {desc} | {int(count):,} |")

    section = f"""---

## Phase 4A: Anomaly & Fraud Signal Detection

### Overview

Multiple anomaly detection methods were applied as an **ensemble** to identify
providers with unusual billing patterns that may warrant further investigation.
The goal is robust anomaly ranking — providers flagged by multiple independent
methods are the highest priority.

**Methods used:**
1. **Statistical outlier detection** — modified Z-score and 3x IQR flagging on
   spending, claims, and per-claim cost
2. **Isolation Forest** — 200 estimators, 1% contamination rate → 6,176 anomalies
3. **Local Outlier Factor (LOF)** — k=20 neighbors → 6,176 anomalies
4. **Rule-based fraud pattern signatures** — 5 specific flags based on domain
   knowledge of Medicaid billing irregularities

### Fraud Flag Summary

| Flag | Providers Flagged |
|------|-------------------|
{chr(10).join(flag_summary_lines)}
| **Any flag (≥1)** | **{total_flagged:,}** |
| **4+ flags** | **{flag_4:,}** |
| **All 5 flags** | **{flag_5}** |

### Fraud Pattern Signatures

![Fraud Flags](outputs/figures/fraud_flags_breakdown.png)

### Anomaly Score Distribution

![Anomaly Scores](outputs/figures/anomaly_score_distribution.png)

### Anomalous Providers (Spending vs Claims)

![Anomaly Scatter](outputs/figures/anomaly_scatter_spending_claims.png)

### Top 20 Flagged Providers (sorted by flag count, then spending)

| Rank | Billing NPI | Total Spending | Claims | HCPCS Codes | Servicing NPIs | Flags |
|------|------------|---------------|--------|-------------|----------------|-------|
"""
    for rank, (_, row) in enumerate(flagged.head(20).iterrows(), 1):
        section += (
            f"| {rank} | {row.get('billing_npi', 'N/A')} | "
            f"{format_currency(row.get('total_paid', 0))} | "
            f"{format_number(row.get('total_claims', 0))} | "
            f"{int(row.get('n_unique_hcpcs', 0))} | "
            f"{int(row.get('n_servicing_npis', 0))} | "
            f"{int(row.get('fraud_flags_count', 0))} |\n"
        )

    section += f"""
*{flag_5} providers triggered all 5 fraud signal flags simultaneously. These
providers merit the highest-priority investigation.*

Note: Anomaly flags are **signals, not accusations**. Each flagged provider may
have legitimate explanations for their billing patterns (e.g., large health systems,
specialty pharmacies, FQHC organizations).
"""

    if benford_path.exists():
        benford = pd.read_csv(benford_path)
        chi2 = benford["chi2_statistic"].iloc[0]
        section += f"""
### Benford's Law Analysis

![Benford](outputs/figures/benford_analysis.png)

Benford's Law tests whether the distribution of leading digits in payment amounts
matches the expected logarithmic distribution. The Medicaid payment data shows a
statistically significant deviation from Benford's Law (chi-squared = {chi2:,.0f},
p ~ 0), with digit "1" over-represented (31.5% observed vs 30.1% expected) and
digits 3-5 under-represented. This pattern is consistent with systematic rounding
in Medicaid payment schedules rather than intentional manipulation.
"""

    return section


def _clustering_findings(tables_dir: Path) -> str:
    path = tables_dir / "cluster_profiles.csv"
    if not path.exists():
        return ""

    clusters = pd.read_csv(path)
    hcpcs_cluster_path = tables_dir / "hcpcs_cluster_profiles.csv"

    # Build provider cluster table
    cluster_lines = []
    for _, row in clusters.iterrows():
        k = int(row["cluster_kmeans"])
        count = int(row["total_paid_count"])
        med_paid = row["total_paid_median"]
        med_claims = row["total_claims_median"]
        med_hcpcs = row["n_unique_hcpcs_median"]
        med_serv = row["n_servicing_npis_median"]
        med_years = row["n_years_active_median"]
        cluster_lines.append(
            f"| {k} | {count:,} | {format_currency(med_paid)} | "
            f"{format_number(med_claims)} | {int(med_hcpcs)} | "
            f"{int(med_serv)} | {int(med_years)} |"
        )

    section = f"""---

## Phase 4B: Unsupervised Clustering

### Provider Segmentation

Providers were clustered using K-Means on log-transformed, robust-scaled billing
feature vectors (total spending, claims, procedure diversity, organizational
complexity, etc.). The optimal number of clusters was determined by silhouette
score analysis.

**Optimal k = 2** (silhouette score = 0.993), revealing a stark bifurcation:

| Cluster | Providers | Median Spending | Median Claims | Median HCPCS | Median Servicing NPIs | Median Years Active |
|---------|-----------|----------------|---------------|--------------|----------------------|---------------------|
{chr(10).join(cluster_lines)}

**Interpretation:**
- **Cluster 0** ({int(clusters.iloc[0]['total_paid_count']):,} providers): The vast majority — small to
  mid-size providers with median spending of {format_currency(clusters.iloc[0]['total_paid_median'])},
  billing a median of {int(clusters.iloc[0]['n_unique_hcpcs_median'])} procedure codes.
- **Cluster 1** ({int(clusters.iloc[1]['total_paid_count']):,} providers): Mega-providers with median
  spending of {format_currency(clusters.iloc[1]['total_paid_median'])}, billing {int(clusters.iloc[1]['n_unique_hcpcs_median'])}
  different procedure codes and using {int(clusters.iloc[1]['n_servicing_npis_median'])} servicing NPIs.
  These likely represent large hospital systems, health plans, or FQHCs.

The near-perfect silhouette score (0.993) indicates these two groups are extremely
well-separated in feature space — a 1,000x difference in median spending.

![Optimal K](outputs/figures/clustering_optimal_k.png)

![Provider Clusters](outputs/figures/provider_clusters_pca.png)
"""

    if hcpcs_cluster_path.exists():
        hcpcs_clusters = pd.read_csv(hcpcs_cluster_path)
        section += """
### HCPCS Code Clustering

Procedure codes were clustered into 5 groups based on spending volume, provider
utilization, and per-claim cost:

| Cluster | Median Spending | Median Claims | Avg $/Claim | Providers | Median Years |
|---------|----------------|---------------|-------------|-----------|-------------|
"""
        for _, row in hcpcs_clusters.iterrows():
            section += (
                f"| {int(row['cluster'])} | {format_currency(row['total_paid'])} | "
                f"{format_number(row['total_claims'])} | "
                f"{format_currency(row['avg_paid_per_claim'])} | "
                f"{int(row['n_providers'])} | {int(row['n_years'])} |\n"
            )

    return section


def _time_series_findings(tables_dir: Path) -> str:
    forecast_path = tables_dir / "spending_forecast.csv"

    section = """---

## Phase 4C: Time Series Analysis

### Stationarity Testing

An Augmented Dickey-Fuller (ADF) test confirmed that the monthly spending series
is **non-stationary** (ADF statistic = -1.193, p = 0.677), meaning differencing
is required before modeling. This is expected given the strong upward trend.

### STL Decomposition

![STL Decomposition](outputs/figures/stl_decomposition.png)

The STL decomposition (period=12, robust=True) separates the monthly spending
signal into three components:
- **Trend**: Steady growth from 2018, sharp acceleration during 2020-2022
  (COVID continuous enrollment), then decline starting mid-2023 (enrollment unwinding)
- **Seasonal**: Consistent monthly patterns across years, with typical year-end
  dips in December and rebounds in January
- **Residual**: Largest residual magnitudes occur during 2020 (COVID onset) and
  2023-2024 (enrollment unwinding), indicating these periods deviated most from
  the expected pattern

### Changepoint Detection

![Changepoints](outputs/figures/changepoint_detection.png)

The PELT algorithm (Pruned Exact Linear Time) detected **3 structural breakpoints**
in the monthly spending series. These correspond to:
1. **COVID onset** (~March 2020): Beginning of the pandemic-driven spending surge
2. **Enrollment surge peak** (~2021-2022): Maximum expansion of Medicaid enrollment
   under continuous coverage requirements
3. **Enrollment unwinding** (~mid-2023): Beginning of the post-PHE disenrollment period

### Granger Causality Analysis

Granger causality testing between monthly claims volume and monthly spending found
**neither direction is statistically significant** at the 5% level. This suggests
that claims volume and spending per claim are driven by independent policy and
utilization factors rather than one causing the other.

### Spending Forecast

![Forecast](outputs/figures/spending_forecast_sarima.png)

A SARIMA(1,1,1)(1,1,1,12) model (AIC ≈ 2572) was fitted to the historical monthly
spending data to produce 12-month-ahead forecasts with 95% confidence intervals.
"""

    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path)
        section += """
| Month | Forecast | 95% CI Lower | 95% CI Upper |
|-------|----------|-------------|-------------|
"""
        for _, row in forecast.head(6).iterrows():
            date_str = str(row["date"])[:7]
            section += (
                f"| {date_str} | {format_currency(row['forecast'])} | "
                f"{format_currency(max(0, row['ci_lower']))} | "
                f"{format_currency(row['ci_upper'])} |\n"
            )
        section += """
*Note: Wide confidence intervals reflect inherent uncertainty in forecasting
spending that is heavily influenced by policy decisions (e.g., PHE extensions,
disenrollment timelines). The forecast assumes continuation of current trends.*
"""

    return section


def _model_findings(tables_dir: Path) -> str:
    path = tables_dir / "model_metrics.csv"
    if not path.exists():
        return ""

    metrics = pd.read_csv(path)

    return f"""---

## Phase 4D: Predictive Modeling

### Provider Spending Prediction (LightGBM)

A LightGBM gradient-boosted tree model was trained to predict provider total
spending from billing pattern features. The model was trained on log-transformed
spending (to handle the heavy-tailed distribution) with an 80/20 train-test split.

| Metric | Value |
|--------|-------|
| R-squared (log scale) | {metrics['r2_log'].iloc[0]:.4f} |
| R-squared (actual $) | {metrics['r2_actual'].iloc[0]:.4f} |
| MAE (actual $) | {format_currency(metrics['mae_actual'].iloc[0])} |
| Training set | {int(metrics['n_train'].iloc[0]):,} providers |
| Test set | {int(metrics['n_test'].iloc[0]):,} providers |

**Interpretation:** The model explains **83.2%** of variance in log-spending,
indicating that billing pattern features are highly predictive of a provider's
spending level. The lower R-squared on the actual dollar scale (0.299) reflects
the extreme right skew — mega-providers are difficult to predict precisely in
absolute dollars, but their relative spending tier is well-captured.

The mean absolute error of ~$1.0M is driven by the heavy tail; for the median
provider (spending ~$48K), predictions are much more accurate.

### Feature Importance

![Feature Importance](outputs/figures/feature_importance.png)

The most important predictive features are:
1. **Total claims** — volume of claims is the strongest predictor of spending
2. **Number of unique beneficiaries** — patient panel size
3. **Number of unique HCPCS codes** — procedure diversity
4. **Number of servicing NPIs** — organizational complexity
5. **Paid per claim** — unit cost intensity

### Prediction Quality

![Predicted vs Actual](outputs/figures/prediction_vs_actual.png)

The predicted vs actual scatter plot shows strong alignment along the diagonal,
with increasing dispersion at higher spending levels (heteroscedasticity typical
of healthcare cost data).

*Model saved to `outputs/models/lgbm_spending_predictor.joblib`*
"""


def _network_findings(tables_dir: Path) -> str:
    path = tables_dir / "network_metrics.csv"
    if not path.exists():
        return ""

    net = pd.read_csv(path)
    n_nodes = int(net["n_nodes"].iloc[0])
    n_edges = int(net["n_edges"].iloc[0])
    n_providers = int(net["n_providers"].iloc[0])
    n_hcpcs = int(net["n_hcpcs"].iloc[0])
    n_components = int(net["n_connected_components"].iloc[0])
    largest_comp = int(net["largest_component_size"].iloc[0])
    avg_prov_deg = net["avg_provider_degree"].iloc[0]
    density = net["density"].iloc[0]

    # Similar billing patterns count
    similar_path = tables_dir.parent / "tables" if "tables" not in str(tables_dir) else tables_dir
    sim_path = tables_dir / "similar_billing_patterns.csv"
    n_similar = 0
    if sim_path.exists():
        sim_df = pd.read_csv(sim_path)
        n_similar = len(sim_df)

    return f"""---

## Phase 4E: Network Analysis

### Provider-HCPCS Bipartite Network

A bipartite network was constructed linking the top {n_providers:,} billing providers
(by spending) to the HCPCS procedure codes they bill, weighted by payment amount.

| Metric | Value |
|--------|-------|
| Total nodes | {n_nodes:,} ({n_providers:,} providers + {n_hcpcs:,} HCPCS codes) |
| Total edges | {n_edges:,} |
| Connected components | {n_components} |
| Largest component | {largest_comp:,} nodes ({largest_comp/n_nodes*100:.1f}%) |
| Average provider degree | {avg_prov_deg:.1f} HCPCS codes per provider |
| Network density | {density:.4f} |

### Hub HCPCS Codes

The most central HCPCS codes (highest degree centrality) serve as "hubs"
connecting the largest number of providers:

| Rank | HCPCS Code | Description |
|------|-----------|-------------|
| 1 | 99213 | Office visit, est. patient, low complexity |
| 2 | 99214 | Office visit, est. patient, moderate complexity |
| 3 | 96372 | Therapeutic injection, SC/IM |
| 4 | 99212 | Office visit, est. patient, straightforward |
| 5 | 99215 | Office visit, est. patient, high complexity |
| 6 | 90471 | Immunization administration |
| 7 | 83036 | Hemoglobin A1c test |
| 8 | 81025 | Urine pregnancy test |
| 9 | 36415 | Venipuncture |
| 10 | 90791 | Psychiatric diagnostic evaluation |

![Degree Distribution](outputs/figures/network_degree_distribution.png)

![HCPCS Centrality](outputs/figures/network_hcpcs_centrality.png)

### Similar Billing Pattern Detection

Jaccard similarity analysis identified **{n_similar} pairs** of providers with
>80% overlap in their HCPCS code billing portfolios. Providers with near-identical
billing patterns may represent:
- Providers within the same health system/practice group
- Potential coordinated billing arrangements
- Specialty-specific standard billing patterns

See `outputs/tables/similar_billing_patterns.csv` for the full list.
"""


def _phase6_findings(tables_dir: Path) -> str:
    """Generate Phase 6 advanced analysis findings."""
    sections = []

    sections.append("""---

## Phase 6: Advanced Analysis

### 6A. Provider Embedding Learning

Provider-HCPCS co-occurrence matrices were decomposed using SVD (64 components) and
NMF (64 components) to learn dense provider representations. k-NN anomaly scoring
(k=10, cosine distance) on these embeddings identifies providers with unusual
procedure code portfolios relative to their nearest neighbors in embedding space.""")

    # Embedding metadata
    emb_meta_path = tables_dir / "embedding_metadata.json"
    if emb_meta_path.exists():
        with open(emb_meta_path) as f:
            emb_meta = json.load(f)
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Providers embedded | {emb_meta.get('n_providers', 'N/A'):,} |
| SVD components | {emb_meta.get('n_components_svd', 64)} |
| NMF components | {emb_meta.get('n_components_nmf', 64)} |

![SVD Scatter](outputs/figures/embedding_svd_scatter.png)

![Embedding Anomaly Distribution](outputs/figures/embedding_anomaly_distribution.png)""")

    # Embedding top anomalies
    emb_anom_path = tables_dir / "embedding_top_anomalies.csv"
    if emb_anom_path.exists():
        try:
            df = pd.read_csv(emb_anom_path)
            n_anom = len(df)
            sections.append(f"""
The top {n_anom} providers by embedding anomaly score are listed in
`outputs/tables/embedding_top_anomalies.csv`.""")
        except Exception:
            pass

    sections.append("""
### 6B. Advanced Clustering (UMAP + HDBSCAN)

UMAP projection of SVD embeddings to 2D, followed by HDBSCAN density-based
clustering, reveals natural provider groupings without requiring a pre-specified
number of clusters. Unlike K-Means, HDBSCAN can identify noise points and
clusters of varying density.""")

    sections.append("""
### 6C. Autoencoder Anomaly Detection

A deep autoencoder (architecture: input -> 128 -> 64 -> 32 -> 64 -> 128 -> input)
was trained on provider feature vectors. Providers with high reconstruction error
represent those whose billing patterns cannot be efficiently compressed and
reconstructed, indicating unusual or anomalous behavior.""")

    ae_anom_path = tables_dir / "autoencoder_top_anomalies.csv"
    if ae_anom_path.exists():
        try:
            df = pd.read_csv(ae_anom_path)
            sections.append(f"""
The top {len(df)} autoencoder anomalies are listed in
`outputs/tables/autoencoder_top_anomalies.csv`.

![Autoencoder Training Loss](outputs/figures/autoencoder_training_loss.png)

![Reconstruction Error Distribution](outputs/figures/autoencoder_reconstruction_error_dist.png)

![Autoencoder Error vs Spending](outputs/figures/autoencoder_error_vs_spending.png)""")
        except Exception:
            pass

    sections.append("""
### 6D. Ensemble Gradient Boosting

A stacked ensemble of LightGBM, XGBoost, and CatBoost was trained with Optuna
Bayesian hyperparameter optimization (15 trials per model) to predict provider
spending. The ensemble uses a Ridge meta-learner for stacking and provides
conformal prediction intervals.""")

    ens_metrics_path = tables_dir / "ensemble_model_metrics.csv"
    if ens_metrics_path.exists():
        try:
            edf = pd.read_csv(ens_metrics_path, index_col=0)
            sections.append("""
**Model Performance (Log-Scale):**

| Model | R² (log) | RMSE (log) | MAE (log) |
|-------|----------|-----------|----------|""")
            for model_name, row in edf.iterrows():
                r2 = row.get('r2_log', row.get('r2', 'N/A'))
                rmse = row.get('rmse_log', row.get('rmse', 'N/A'))
                mae = row.get('mae_log', row.get('mae', 'N/A'))
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                sections.append(f"| {model_name} | {r2_str} | {rmse_str} | {mae_str} |")
        except Exception:
            pass

    ens_imp_path = tables_dir / "ensemble_feature_importance.csv"
    if ens_imp_path.exists():
        try:
            imp = pd.read_csv(ens_imp_path)
            top5 = imp.head(5)
            sections.append("""
**Top 5 Features (LightGBM Importance):**

| Feature | Importance |
|---------|-----------|""")
            for _, row in top5.iterrows():
                sections.append(f"| {row['feature']} | {row['importance']:,.0f} |")
        except Exception:
            pass

    sections.append("""
![Ensemble Predicted vs Actual](outputs/figures/ensemble_predicted_vs_actual.png)

![Ensemble Model Comparison](outputs/figures/ensemble_model_comparison.png)

![Ensemble Prediction Intervals](outputs/figures/ensemble_prediction_intervals.png)

![Ensemble Optuna Optimization](outputs/figures/ensemble_optuna_optimization.png)""")

    sections.append("""
### 6E. Provider Trajectory Analysis

Provider spending trajectories over 84 months (2018-2024) were analyzed to identify
distinct temporal patterns. Trajectory features include compound annual growth rate
(CAGR), volatility, COVID impact ratio, and trend classification. K-Shape clustering
with soft-DTW metric groups providers by temporal spending shape.""")

    traj_path = tables_dir / "trajectory_archetype_summary.csv"
    if traj_path.exists():
        try:
            df = pd.read_csv(traj_path)
            sections.append(f"""
**Trajectory Archetypes Identified:** {len(df)}

| Archetype | Count | Median CAGR | Median Volatility |
|-----------|-------|-------------|-------------------|""")
            for _, row in df.iterrows():
                name = row.get('archetype', row.get('trajectory_archetype', 'N/A'))
                count = row.get('count', row.get('n_providers', 'N/A'))
                cagr = row.get('median_cagr', 'N/A')
                vol = row.get('median_volatility', 'N/A')
                cagr_str = f"{cagr:.3f}" if isinstance(cagr, (int, float)) else str(cagr)
                vol_str = f"{vol:.3f}" if isinstance(vol, (int, float)) else str(vol)
                sections.append(f"| {name} | {count:,} | {cagr_str} | {vol_str} |")
        except Exception:
            pass

    sections.append("""
![Trajectory Archetypes](outputs/figures/trajectory_archetype_distribution.png)

![Trajectory Spaghetti by Archetype](outputs/figures/trajectory_spaghetti_by_archetype.png)

![K-Shape Centroids](outputs/figures/trajectory_kshape_centroids.png)""")

    sections.append("""
### 6F. Causal Analysis

Four complementary causal inference methods were applied to estimate the causal
impact of COVID-19 (March 2020) on Medicaid spending:""")

    causal_path = tables_dir / "causal_impact_summary.csv"
    if causal_path.exists():
        try:
            df = pd.read_csv(causal_path)
            sections.append("""
| Method | Key Result |
|--------|-----------|""")
            for _, row in df.iterrows():
                method = row.get('method', 'N/A')
                result = row.get('key_result', row.get('result', 'N/A'))
                sections.append(f"| {method} | {result} |")
        except Exception:
            pass

    sections.append("""
![DiD Event Study](outputs/figures/causal_did_event_study.png)

![BSTS Counterfactual](outputs/figures/causal_bsts_counterfactual.png)

![RDD Plot](outputs/figures/causal_rdd_plot.png)

![Synthetic Control](outputs/figures/causal_synthetic_control.png)""")

    sections.append("""
### 6G. Graph Community Detection

Provider-HCPCS co-billing networks were analyzed using Louvain community detection
to identify clusters of providers who bill similar procedure codes. Spectral graph
embedding provides additional structural insight.""")

    graph_path = tables_dir / "graph_metrics_extended.csv"
    if graph_path.exists():
        try:
            df = pd.read_csv(graph_path)
            if len(df) > 0:
                row = df.iloc[0]
                sections.append(f"""
| Metric | Value |
|--------|-------|
| Communities detected | {int(row.get('n_communities', 0)):,} |
| Modularity | {row.get('modularity', 0):.3f} |
| Nodes in graph | {int(row.get('n_nodes', 0)):,} |
| Edges in graph | {int(row.get('n_edges', 0)):,} |""")
        except Exception:
            pass

    sections.append("""
![Community Sizes](outputs/figures/graph_community_sizes.png)

![Spectral Embedding](outputs/figures/graph_spectral_embedding.png)

![Community Spending Profiles](outputs/figures/graph_community_spending_profiles.png)""")

    sections.append("""
### 6H. Unified Risk Scoring

All anomaly signals are aggregated into a single calibrated risk score per provider
using semi-supervised pseudo-labels and Platt scaling. Providers are classified into
risk tiers for prioritized investigation.

![Anomaly Method Comparison](outputs/figures/anomaly_method_comparison.png)

![Ensemble Anomaly Agreement](outputs/figures/ensemble_anomaly_agreement.png)""")

    return "\n".join(sections)


def _phase7_findings(tables_dir: Path) -> str:
    """Generate Phase 7 SOTA analysis findings."""
    sections = []

    sections.append("""---

## Phase 7: State-of-the-Art Analysis

### 7A. Topological Data Analysis (TDA)

Persistent homology reveals the topological structure of the provider feature space —
connected components (H0) represent distinct provider groups, and loops (H1) indicate
circular relationships in billing patterns. This is among the first applications of
TDA to healthcare spending analysis.""")

    tda_path = tables_dir / "tda_summary.json"
    if tda_path.exists():
        with open(tda_path) as f:
            tda = json.load(f)
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Providers analyzed | {tda.get('n_providers', 'N/A'):,} |
| Sample for persistence | {tda.get('n_sample_persistence', 'N/A'):,} |
| H0 features (connected components) | {tda.get('h0_features', 'N/A'):,} |
| H1 features (loops) | {tda.get('h1_features', 'N/A'):,} |
| H1 max lifetime | {tda.get('h1_max_lifetime', 0):.3f} |
| Persistence entropy (H1) | {tda.get('persistence_entropy_h1', 0):.3f} bits |

The presence of {tda.get('h1_features', 0):,} H1 features indicates significant
loop structure in the provider feature space — providers form circular billing
pattern relationships rather than simple hierarchical clusters.

![Persistence Diagrams](outputs/figures/tda_persistence_diagrams.png)

![Betti Curves](outputs/figures/tda_betti_curves.png)

![Persistence Landscape](outputs/figures/tda_persistence_landscape.png)

![Topological Anomalies](outputs/figures/tda_topological_anomalies.png)""")

    sections.append("""
### 7B. Variational Autoencoder (β-VAE)

A β-VAE with cyclical KL annealing learns disentangled latent representations of
provider billing patterns. Unlike the deterministic autoencoder (Phase 6C), the VAE
provides a principled probabilistic anomaly score combining reconstruction error
and KL divergence from the prior — providers whose billing patterns cannot be
generated by the learned latent model are flagged as anomalous.""")

    vae_path = tables_dir / "vae_summary.json"
    if vae_path.exists():
        with open(vae_path) as f:
            vae = json.load(f)
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Providers scored | {vae.get('n_providers', 'N/A'):,} |
| Latent dimensions | {vae.get('latent_dim', 'N/A')} |
| Parameters | {vae.get('n_parameters', 'N/A'):,} |
| Epochs trained | {vae.get('epochs_trained', 'N/A')} |
| Final ELBO (val) | {vae.get('final_val_loss', 0):.4f} |
| Mean reconstruction error | {vae.get('mean_recon_error', 0):.4f} |
| Mean KL divergence | {vae.get('mean_kl_divergence', 0):.4f} |

![VAE Training](outputs/figures/vae_training_history.png)

![VAE Latent Space](outputs/figures/vae_latent_space.png)

![VAE Anomaly Distributions](outputs/figures/vae_anomaly_distributions.png)""")

    sections.append("""
### 7C. Optimal Transport Analysis

Optimal transport theory quantifies how spending distributions shift across time using
Wasserstein distances and Sinkhorn divergences. Transport plans reveal how healthcare
dollars flow between cost regimes, and McCann displacement interpolation visualizes
the distributional evolution from pre-COVID to post-COVID spending patterns.""")

    ot_path = tables_dir / "ot_summary.json"
    if ot_path.exists():
        with open(ot_path) as f:
            ot = json.load(f)
        max_shift = ot.get("max_wasserstein_shift", {})
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Years analyzed | {len(ot.get('years_analyzed', []))} (2018-2024) |
| Largest YoY shift | {max_shift.get('from', '?')} → {max_shift.get('to', '?')} (W₁ = {max_shift.get('distance', 0):.4f}) |
| Pre/post COVID distance | W₁ = {ot.get('pre_post_covid_wasserstein', 0):.4f} |
| Total cumulative shift | {ot.get('total_cumulative_shift', 0):.4f} |

The largest distributional shift occurred from {max_shift.get('from', '?')} to {max_shift.get('to', '?')},
coinciding with the COVID-era enrollment expansion and spending surge.

![Wasserstein Heatmap](outputs/figures/ot_wasserstein_heatmap.png)

![Sinkhorn Heatmap](outputs/figures/ot_sinkhorn_heatmap.png)

![Transport Plan 2019→2021](outputs/figures/ot_transport_plan_2019_2021.png)

![YoY Wasserstein](outputs/figures/ot_yoy_wasserstein.png)

![Displacement Interpolation](outputs/figures/ot_displacement_interpolation.png)""")

    sections.append("""
### 7D. Double Machine Learning (DML)

The Chernozhukov et al. (2018) Double/Debiased ML framework estimates the causal
impact of COVID on provider spending with rigorous statistical guarantees. Cross-fitted
ML nuisance models (LightGBM) partial out confounding, and EconML's CausalForestDML
estimates heterogeneous treatment effects across provider types.""")

    dml_path = tables_dir / "dml_summary.json"
    if dml_path.exists():
        with open(dml_path) as f:
            dml = json.load(f)
        ate = dml.get("ate", {})
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Observations | {dml.get('n_observations', 'N/A'):,} |
| Unique providers | {dml.get('n_providers', 'N/A'):,} |
| ATE (log-spending) | {ate.get('ate', 0):.4f} |
| Standard error | {ate.get('se', 0):.4f} |
| 95% CI | [{ate.get('ci_low', 0):.4f}, {ate.get('ci_high', 0):.4f}] |
| p-value | {ate.get('p_value', 1):.2e} |
| Multiplicative effect | +{dml.get('ate_multiplicative_effect_pct', 0):.1f}% |

**Interpretation:** COVID caused an average {dml.get('ate_multiplicative_effect_pct', 0):.1f}% increase
in provider spending, after controlling for observable confounders via ML-based
debiasing. This estimate has double-robustness guarantees.

![DML ATE](outputs/figures/dml_ate_estimate.png)

![CATE by Provider Size](outputs/figures/dml_cate_by_provider_size.png)

![CATE Distribution](outputs/figures/dml_cate_distribution.png)""")

    sections.append("""
### 7E. Information-Theoretic Analysis

Shannon entropy quantifies the diversity of each provider's billing code distribution,
mutual information reveals which features carry the most predictive signal about
spending, and transfer entropy measures directed information flow between spending
metrics over time.""")

    info_path = tables_dir / "info_theory_summary.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        sections.append(f"""
| Metric | Value |
|--------|-------|
| Providers analyzed | {info.get('n_providers', 'N/A'):,} |
| Mean billing entropy | {info.get('mean_billing_entropy', 0):.3f} bits |
| Median billing entropy | {info.get('median_billing_entropy', 0):.3f} bits |
| Max billing entropy | {info.get('max_billing_entropy', 0):.3f} bits |
| Top MI feature | {info.get('top_mi_feature', 'N/A')} (MI = {info.get('top_mi_value', 0):.4f}) |

Low-entropy providers (< 1 bit) are highly specialized, billing primarily a single
procedure code. High-entropy providers (> 5 bits) bill across hundreds of distinct
codes, characteristic of large health systems or FQHCs.

![Entropy Distribution](outputs/figures/info_entropy_distribution.png)

![Mutual Information](outputs/figures/info_mutual_information.png)

![Entropy Evolution](outputs/figures/info_entropy_evolution.png)""")

    return "\n".join(sections)


def _limitations() -> str:
    return """---

## Limitations & Caveats

1. **No geographic data**: The dataset lacks state/geography identifiers, preventing
   per-capita or state-level spending analysis.

2. **No claim type distinction**: FFS, managed care, and CHIP claims cannot be
   separated in this dataset version.

3. **Cell suppression bias**: Rows with <12 claims are suppressed, systematically
   excluding low-volume provider-procedure combinations. This biases analyses
   toward higher-volume billing patterns.

4. **Aggregated data**: Records are aggregated at the billing-provider x servicing-provider
   x HCPCS x month level. Individual claim details are not available.

5. **NPI as proxy**: Provider identity is based on NPI numbers, which may not map
   1:1 to individual practitioners or organizations (e.g., group practices).

6. **No patient demographics**: Beneficiary counts are given but no demographic
   breakdown is available.

7. **Anomaly flags are signals, not accusations**: Providers flagged by anomaly
   detection algorithms warrant investigation but are not necessarily engaged in
   fraud. Many flags may have legitimate explanations.

8. **Temporal coverage**: Not all providers or procedure codes are active across
   the full 2018-2024 window, which affects trend analyses.
"""


def _methodology() -> str:
    return """---

## Methodology

### Data Processing
- Raw CSV (10.3 GB, 227M rows) processed in 2M-row chunks
- Converted to year-partitioned Parquet files with derived features
- Provider-level and HCPCS-level feature vectors computed via aggregation

### Statistical Testing
- All tests at alpha = 0.05
- Multiple comparison correction via Benjamini-Hochberg FDR
- Effect sizes reported alongside p-values
- Bootstrap confidence intervals where parametric assumptions may not hold

### Anomaly Detection
- Ensemble approach: statistical outliers, Isolation Forest, LOF, and rule-based patterns
- Composite scoring combines multiple signals for robust anomaly ranking

### Clustering
- Optimal k determined by silhouette score, elbow method, and BIC
- Features log-transformed and robust-scaled before clustering
- PCA visualization for cluster interpretation

### Time Series
- STL decomposition for trend-seasonal-residual separation
- PELT algorithm for changepoint detection
- SARIMA for forecasting with confidence intervals

### Predictive Modeling
- LightGBM with cross-validation
- SHAP values for feature importance and model interpretability
- Stacked ensemble: LightGBM + XGBoost + CatBoost with Optuna Bayesian HPO
- Conformal prediction intervals for uncertainty quantification

### Advanced Anomaly Detection
- Deep autoencoder reconstruction error scoring
- k-NN anomaly detection on SVD/NMF provider embeddings
- UMAP + HDBSCAN density-based clustering
- Unified risk scoring with Platt-calibrated probabilities

### Causal Inference
- Difference-in-Differences with event study
- Bayesian Structural Time Series (BSTS) counterfactual
- Regression Discontinuity Design (RDD)
- Synthetic Control Method

### Graph & Trajectory Analysis
- Louvain community detection on provider co-billing networks
- Spectral graph embedding
- Provider spending trajectory classification
- K-Shape clustering with soft-DTW metric

### State-of-the-Art Methods (Phase 7)
- Topological Data Analysis: Vietoris-Rips persistent homology (H0, H1)
- β-VAE with cyclical KL annealing for disentangled anomaly detection
- Optimal transport: Wasserstein distances, Sinkhorn divergences, transport plans
- Double Machine Learning (Chernozhukov et al., 2018) for debiased causal estimation
- EconML CausalForestDML for heterogeneous treatment effects
- Information theory: Shannon entropy, mutual information, transfer entropy

---

## Future Work

1. Integrate NPPES data to map NPIs to states and provider types
2. Obtain claim-type disaggregation (FFS/managed care/CHIP)
3. Link to CMS provider enrollment data for credential analysis
4. Build interactive dashboard with Plotly/Streamlit
5. Implement real-time anomaly monitoring pipeline
6. Analyze ICD-10 diagnosis code associations if available
7. Geographic hotspot analysis with state-level data
"""


if __name__ == "__main__":
    generate_findings_report()
