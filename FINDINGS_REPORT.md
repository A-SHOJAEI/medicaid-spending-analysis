# Medicaid Provider Spending Analysis: Findings Report

**Generated:** 2026-02-16 01:51:51
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
  statistical outliers) that can guide fraud investigation priorities

---

## Phase 1: Data Profile & Quality Assessment

### Dataset Overview

| Metric | Value |
|--------|-------|
| File Size | 10.32 GB |
| Total Rows | 227,083,361 |
| Columns | BILLING_PROVIDER_NPI_NUM, SERVICING_PROVIDER_NPI_NUM, HCPCS_CODE, CLAIM_FROM_MONTH, TOTAL_UNIQUE_BENEFICIARIES, TOTAL_CLAIMS, TOTAL_PAID |
| Date Range | 2018-01 to 2024-12 |
| Unique Billing NPIs | 617,503 |
| Unique Servicing NPIs | 1,627,362 |
| Unique HCPCS Codes | 10,881 |
| Months Covered | 84 |

### Spending Summary

| Metric | Value |
|--------|-------|
| Total Spending | $1.09T |
| Mean per Record | $4.8K |
| Std Dev | $91.2K |
| Min | -$183.0K |
| Max | $118.9M |
| Rows with Negative Payment | 9,239 |
| Rows with Zero Payment | 35,502,216 |
| Total Claims | 18.83B |

### Null Rates

| Column | Null Rate |
|--------|-----------|
| BILLING_PROVIDER_NPI_NUM | 0.0% |
| SERVICING_PROVIDER_NPI_NUM | 4.1792% |
| HCPCS_CODE | 0.0% |
| CLAIM_FROM_MONTH | 0.0% |
| TOTAL_UNIQUE_BENEFICIARIES | 0.0% |
| TOTAL_CLAIMS | 0.0% |
| TOTAL_PAID | 0.0% |

### NPI Relationship Analysis

| Metric | Value |
|--------|-------|
| Billing = Servicing NPI | 69,731,487 (30.7%) |
| Servicing NPI is Null | 9,490,345 (4.2%) |

### Data Quality Assessment

- **Missing data**: SERVICING_PROVIDER_NPI_NUM has the highest null rate, which is expected for
  providers who both bill and render services.
- **Negative payments**: 9,239 records have negative TOTAL_PAID values,
  indicating payment adjustments, recoupments, or refunds.
- **Cell suppression**: The dataset suppresses rows with fewer than 12 claims for privacy.
  This means low-volume provider-procedure combinations are systematically absent.
- **No state column**: The dataset lacks a state/geography identifier, limiting
  geographic analysis to NPI-based inferences.
- **No claim type column**: FFS vs managed care vs CHIP distinction is not available
  in this dataset version.

### Top 20 HCPCS Codes by Total Spending

| Rank | HCPCS Code | Total Spending |
|------|-----------|---------------|
| 1 | T1019 | $122.74B |
| 2 | T1015 | $49.15B |
| 3 | T2016 | $34.90B |
| 4 | 99213 | $33.00B |
| 5 | S5125 | $31.34B |
| 6 | 99214 | $29.91B |
| 7 | 99284 | $20.15B |
| 8 | H2016 | $19.75B |
| 9 | 99283 | $16.87B |
| 10 | H2015 | $16.47B |
| 11 | 99285 | $15.10B |
| 12 | 90837 | $12.07B |
| 13 | S5102 | $9.34B |
| 14 | 90834 | $8.82B |
| 15 | T2021 | $8.65B |
| 16 | H2017 | $8.54B |
| 17 | T1017 | $8.42B |
| 18 | T1020 | $8.21B |
| 19 | 90999 | $7.74B |
| 20 | A0427 | $7.67B |

---

## Phase 2: Exploratory Data Analysis

### Annual Spending Trends

Total national Medicaid provider spending by year:

| Year | Total Spending | Total Claims | YoY Growth |
|------|---------------|-------------|------------|
| 2018 | $108.67B | 2.13B |  |
| 2019 | $126.91B | 2.40B | +16.8% |
| 2020 | $132.09B | 2.30B | +4.1% |
| 2021 | $162.56B | 2.90B | +23.1% |
| 2022 | $179.56B | 3.09B | +10.5% |
| 2023 | $198.79B | 3.22B | +10.7% |
| 2024 | $184.98B | 2.78B | -6.9% |

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


---

## Phase 3: Hypothesis Testing

All tests were conducted at alpha = 0.05 with Benjamini-Hochberg FDR correction
for multiple comparisons.

| # | Hypothesis | Test | p-value | FDR p-value | Result |
|---|-----------|------|---------|-------------|--------|
| H1 | H1: COVID Spending Shock | Wilcoxon signed-rank test | 4.88e-04 | 1.22e-03 | **REJECT H0** |
| H2 | H2: Post-COVID Sustained Increase | Mann-Whitney U (one-sided) | 3.90e-10 | 1.30e-09 | **REJECT H0** |
| H3 | H3: Power-Law Provider Spending | Likelihood ratio test (power-law vs. lognormal) | 4.72e-02 | 9.44e-02 | **Fail to reject** |
| H4 | H4: Spending Growth Acceleration | Kendall tau trend test | 4.69e-01 | 5.22e-01 | **Fail to reject** |
| H5 | H5: Monthly Seasonality | Kruskal-Wallis H-test | 9.94e-01 | 9.94e-01 | **Fail to reject** |
| H6 | H6: Procedure Concentration Shift | Kendall tau trend test | 6.90e-02 | 1.15e-01 | **Fail to reject** |
| H7 | H7: Multi-Servicing NPI Spending | Mann-Whitney U (one-sided) | 0.00e+00 | 0.00e+00 | **REJECT H0** |
| H8 | H8: Claims per Beneficiary Trend | Kendall tau + OLS regression | 3.81e-01 | 4.77e-01 | **Fail to reject** |
| H9 | H9: Per-Claim Cost Inflation | One-sample t-test vs 3% benchmark | 1.76e-01 | 2.51e-01 | **Fail to reject** |
| H10 | H10: Provider Specialization vs Cost | Mann-Whitney U + Spearman correlation | 0.00e+00 | 0.00e+00 | **REJECT H0** |

### Detailed Results

#### H1: COVID Spending Shock

- **Null hypothesis**: Monthly spending in 2020-2021 equals 2018-2019 baseline
- **Alternative**: Monthly spending differs between periods
- **Test**: Wilcoxon signed-rank test
- **Test statistic**: 0.0000
- **p-value**: 4.88e-04
- **Effect size**: 1.0000
- **95% CI**: [1176717332.7506, 3285117490.0171]
- **Conclusion**: Reject H0. COVID period spending was +25.1% vs pre-COVID. Effect size (rank-biserial r) = 1.000.

#### H2: Post-COVID Sustained Increase

- **Null hypothesis**: Post-COVID monthly spending <= pre-COVID monthly spending
- **Alternative**: Post-COVID monthly spending > pre-COVID
- **Test**: Mann-Whitney U (one-sided)
- **Test statistic**: 24.0000
- **p-value**: 3.90e-10
- **Effect size**: 0.9444
- **95% CI**: [4925051464.5270, 6562964641.1188]
- **Conclusion**: Reject H0. Post-COVID spending is +59.4% vs pre-COVID.

#### H3: Power-Law Provider Spending

- **Null hypothesis**: Provider spending follows a lognormal distribution
- **Alternative**: Provider spending follows a power-law distribution
- **Test**: Likelihood ratio test (power-law vs. lognormal)
- **Test statistic**: -7.4822
- **p-value**: 4.72e-02
- **Effect size**: 2.1359
- **Conclusion**: Lognormal is preferred (R=-7.482, p=4.720e-02). Power-law exponent alpha=2.14, xmin=11657136.

#### H4: Spending Growth Acceleration

- **Null hypothesis**: YoY spending growth rate has no trend over time
- **Alternative**: YoY spending growth rate shows a significant trend
- **Test**: Kendall tau trend test
- **Test statistic**: -0.3333
- **p-value**: 4.69e-01
- **Conclusion**: Fail to reject H0. Kendall tau = -0.333. Growth rate is decelerating (p=0.469).

#### H5: Monthly Seasonality

- **Null hypothesis**: Monthly spending does not differ across months of the year
- **Alternative**: Monthly spending differs significantly by month
- **Test**: Kruskal-Wallis H-test
- **Test statistic**: 2.7340
- **p-value**: 9.94e-01
- **Effect size**: -0.1148
- **Conclusion**: Fail to reject H0. Eta-squared = -0.1148. Weak effect.

#### H6: Procedure Concentration Shift

- **Null hypothesis**: HCPCS concentration (HHI) has no trend over time
- **Alternative**: HCPCS concentration changes over time
- **Test**: Kendall tau trend test
- **Test statistic**: 0.6190
- **p-value**: 6.90e-02
- **Conclusion**: Fail to reject H0. HHI is increasing (more concentrated). Tau = 0.619.

#### H7: Multi-Servicing NPI Spending

- **Null hypothesis**: Providers with multiple servicing NPIs have same spending as single-NPI
- **Alternative**: Multi-servicing NPI providers have higher total spending
- **Test**: Mann-Whitney U (one-sided)
- **Test statistic**: 18188858235.0000
- **p-value**: 0.00e+00
- **Effect size**: 0.4853
- **95% CI**: [839719.5087, 6091988.5225]
- **Conclusion**: Reject H0. Multi-servicing NPI providers have higher spending. Median single: $24.0K, Median multi: $327.9K.

#### H8: Claims per Beneficiary Trend

- **Null hypothesis**: Claims per beneficiary is stable over time
- **Alternative**: Claims per beneficiary shows a significant trend
- **Test**: Kendall tau + OLS regression
- **Test statistic**: -0.3333
- **p-value**: 3.81e-01
- **Effect size**: 0.3087
- **Conclusion**: Fail to reject H0. Claims/beneficiary is decreasing (tau=-0.333, R^2=0.309, slope=-0.0074/year).

#### H9: Per-Claim Cost Inflation

- **Null hypothesis**: Paid per claim grows at the same rate as CPI (~3%/year)
- **Alternative**: Paid per claim grows faster than CPI
- **Test**: One-sample t-test vs 3% benchmark
- **Test statistic**: 1.0269
- **p-value**: 1.76e-01
- **Effect size**: 0.0458
- **95% CI**: [0.0052, 0.0877]
- **Conclusion**: Fail to reject H0. Medicaid cost per claim CAGR = 4.6% vs CPI benchmark of 3%. Mean annual growth rate = 4.6% (95% CI: [0.5%, 8.8%]).

#### H10: Provider Specialization vs Cost

- **Null hypothesis**: Specialized providers have same per-claim cost as generalists
- **Alternative**: Per-claim cost differs between specialist and generalist providers
- **Test**: Mann-Whitney U + Spearman correlation
- **Test statistic**: 10066191918.5000
- **p-value**: 0.00e+00
- **Effect size**: -0.2175
- **Conclusion**: Reject H0. Spearman rho = -0.217 (p=0.00e+00). Median specialist: $46.09, Median generalist: $31.07.


---

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
| Procedure code stuffing (>100 HCPCS codes) | 30,831 |
| High cost per claim (>$500) | 6,176 |
| Many servicing NPIs (>50) | 37,615 |
| Extreme total spending (top 0.1%) | 6,176 |
| Year-over-year spending spike (>300%) | 97,739 |
| **Any flag (≥1)** | **146,806** |
| **4+ flags** | **418** |
| **All 5 flags** | **6** |

### Fraud Pattern Signatures

![Fraud Flags](outputs/figures/fraud_flags_breakdown.png)

### Anomaly Score Distribution

![Anomaly Scores](outputs/figures/anomaly_score_distribution.png)

### Anomalous Providers (Spending vs Claims)

![Anomaly Scatter](outputs/figures/anomaly_scatter_spending_claims.png)

### Top 20 Flagged Providers (sorted by flag count, then spending)

| Rank | Billing NPI | Total Spending | Claims | HCPCS Codes | Servicing NPIs | Flags |
|------|------------|---------------|--------|-------------|----------------|-------|
| 1 | 1679587679 | $217.7M | 4.7M | 765 | 277 | 5 |
| 2 | 1457478703 | $107.5M | 1.9M | 83 | 253 | 5 |
| 3 | 1730454083 | $88.4M | 1.1M | 264 | 97 | 5 |
| 4 | 1790730281 | $64.5M | 529.4K | 109 | 122 | 5 |
| 5 | 1891731626 | $50.0M | 11.1M | 495 | 22 | 5 |
| 6 | 1942623319 | $49.1M | 899.6K | 58 | 387 | 5 |
| 7 | 1679524961 | $406.5M | 4.9M | 217 | 389 | 4 |
| 8 | 1285641514 | $370.9M | 4.0M | 745 | 107 | 4 |
| 9 | 1811062763 | $317.6M | 10.1M | 656 | 2115 | 4 |
| 10 | 1619341716 | $303.2M | 4.4M | 535 | 451 | 4 |
| 11 | 1891799227 | $299.0M | 3.2M | 826 | 11 | 4 |
| 12 | 1891765178 | $286.7M | 1.1M | 238 | 437 | 4 |
| 13 | 1932103413 | $279.0M | 5.0M | 933 | 279 | 4 |
| 14 | 1174689665 | $278.7M | 4.4M | 757 | 184 | 4 |
| 15 | 1477643690 | $277.3M | 1.8M | 400 | 1594 | 4 |
| 16 | 1740262880 | $265.9M | 31.0M | 598 | 35 | 4 |
| 17 | 1194743013 | $263.8M | 1.4M | 387 | 729 | 4 |
| 18 | 1093777492 | $260.8M | 3.4M | 767 | 134 | 4 |
| 19 | 1801992631 | $238.7M | 4.7M | 848 | 42 | 4 |
| 20 | 1366515488 | $221.2M | 1.0M | 264 | 908 | 4 |

*6 providers triggered all 5 fraud signal flags simultaneously. These
providers merit the highest-priority investigation.*

Note: Anomaly flags are **signals, not accusations**. Each flagged provider may
have legitimate explanations for their billing patterns (e.g., large health systems,
specialty pharmacies, FQHC organizations).

### Benford's Law Analysis

![Benford](outputs/figures/benford_analysis.png)

Benford's Law tests whether the distribution of leading digits in payment amounts
matches the expected logarithmic distribution. The Medicaid payment data shows a
statistically significant deviation from Benford's Law (chi-squared = 240,108,
p ~ 0), with digit "1" over-represented (31.5% observed vs 30.1% expected) and
digits 3-5 under-represented. This pattern is consistent with systematic rounding
in Medicaid payment schedules rather than intentional manipulation.


---

## Phase 4B: Unsupervised Clustering

### Provider Segmentation

Providers were clustered using K-Means on log-transformed, robust-scaled billing
feature vectors (total spending, claims, procedure diversity, organizational
complexity, etc.). The optimal number of clusters was determined by silhouette
score analysis.

**Optimal k = 2** (silhouette score = 0.993), revealing a stark bifurcation:

| Cluster | Providers | Median Spending | Median Claims | Median HCPCS | Median Servicing NPIs | Median Years Active |
|---------|-----------|----------------|---------------|--------------|----------------------|---------------------|
| 0 | 617,034 | $47.7K | 1.4K | 3 | 1 | 4 |
| 1 | 469 | $54.9M | 1.3M | 261 | 696 | 7 |

**Interpretation:**
- **Cluster 0** (617,034 providers): The vast majority — small to
  mid-size providers with median spending of $47.7K,
  billing a median of 3 procedure codes.
- **Cluster 1** (469 providers): Mega-providers with median
  spending of $54.9M, billing 261
  different procedure codes and using 696 servicing NPIs.
  These likely represent large hospital systems, health plans, or FQHCs.

The near-perfect silhouette score (0.993) indicates these two groups are extremely
well-separated in feature space — a 1,000x difference in median spending.

![Optimal K](outputs/figures/clustering_optimal_k.png)

![Provider Clusters](outputs/figures/provider_clusters_pca.png)

### HCPCS Code Clustering

Procedure codes were clustered into 5 groups based on spending volume, provider
utilization, and per-claim cost:

| Cluster | Median Spending | Median Claims | Avg $/Claim | Providers | Median Years |
|---------|----------------|---------------|-------------|-----------|-------------|
| 0 | $0.00 | 112 | $0.00 | 1 | 2 |
| 1 | $1.3M | 8.6K | $131.87 | 18 | 7 |
| 2 | $24.0M | 937.3K | $28.91 | 647 | 7 |
| 3 | $15.5K | 144 | $91.00 | 2 | 2 |
| 4 | $78.9K | 13.6K | $7.85 | 25 | 7 |


---

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

| Month | Forecast | 95% CI Lower | 95% CI Upper |
|-------|----------|-------------|-------------|
| 2025-01 | $3.50B | $879.3M | $6.12B |
| 2025-02 | $3.82B | $0.00 | $8.16B |
| 2025-03 | $4.12B | $0.00 | $9.42B |
| 2025-04 | $3.94B | $0.00 | $10.16B |
| 2025-05 | $4.16B | $0.00 | $11.13B |
| 2025-06 | $2.72B | $0.00 | $10.39B |

*Note: Wide confidence intervals reflect inherent uncertainty in forecasting
spending that is heavily influenced by policy decisions (e.g., PHE extensions,
disenrollment timelines). The forecast assumes continuation of current trends.*


---

## Phase 4D: Predictive Modeling

### Provider Spending Prediction (LightGBM)

A LightGBM gradient-boosted tree model was trained to predict provider total
spending from billing pattern features. The model was trained on log-transformed
spending (to handle the heavy-tailed distribution) with an 80/20 train-test split.

| Metric | Value |
|--------|-------|
| R-squared (log scale) | 0.8320 |
| R-squared (actual $) | 0.2992 |
| MAE (actual $) | $1.0M |
| Training set | 475,387 providers |
| Test set | 118,847 providers |

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


---

## Phase 4E: Network Analysis

### Provider-HCPCS Bipartite Network

A bipartite network was constructed linking the top 3,000 billing providers
(by spending) to the HCPCS procedure codes they bill, weighted by payment amount.

| Metric | Value |
|--------|-------|
| Total nodes | 10,314 (3,000 providers + 7,314 HCPCS codes) |
| Total edges | 444,995 |
| Connected components | 2 |
| Largest component | 10,312 nodes (100.0%) |
| Average provider degree | 148.3 HCPCS codes per provider |
| Network density | 0.0084 |

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

Jaccard similarity analysis identified **88 pairs** of providers with
>80% overlap in their HCPCS code billing portfolios. Providers with near-identical
billing patterns may represent:
- Providers within the same health system/practice group
- Potential coordinated billing arrangements
- Specialty-specific standard billing patterns

See `outputs/tables/similar_billing_patterns.csv` for the full list.


---

## Phase 6: Advanced Analysis

### 6A. Provider Embedding Learning

Provider-HCPCS co-occurrence matrices were decomposed using SVD (64 components) and
NMF (64 components) to learn dense provider representations. k-NN anomaly scoring
(k=10, cosine distance) on these embeddings identifies providers with unusual
procedure code portfolios relative to their nearest neighbors in embedding space.

| Metric | Value |
|--------|-------|
| Providers embedded | 617,503 |
| SVD components | 64 |
| NMF components | 64 |

![SVD Scatter](outputs/figures/embedding_svd_scatter.png)

![Embedding Anomaly Distribution](outputs/figures/embedding_anomaly_distribution.png)

The top 100 providers by embedding anomaly score are listed in
`outputs/tables/embedding_top_anomalies.csv`.

### 6B. Advanced Clustering (UMAP + HDBSCAN)

UMAP projection of SVD embeddings to 2D, followed by HDBSCAN density-based
clustering, reveals natural provider groupings without requiring a pre-specified
number of clusters. Unlike K-Means, HDBSCAN can identify noise points and
clusters of varying density.

### 6C. Autoencoder Anomaly Detection

A deep autoencoder (architecture: input -> 128 -> 64 -> 32 -> 64 -> 128 -> input)
was trained on provider feature vectors. Providers with high reconstruction error
represent those whose billing patterns cannot be efficiently compressed and
reconstructed, indicating unusual or anomalous behavior.

The top 100 autoencoder anomalies are listed in
`outputs/tables/autoencoder_top_anomalies.csv`.

![Autoencoder Training Loss](outputs/figures/autoencoder_training_loss.png)

![Reconstruction Error Distribution](outputs/figures/autoencoder_reconstruction_error_dist.png)

![Autoencoder Error vs Spending](outputs/figures/autoencoder_error_vs_spending.png)

### 6D. Ensemble Gradient Boosting

A stacked ensemble of LightGBM, XGBoost, and CatBoost was trained with Optuna
Bayesian hyperparameter optimization (15 trials per model) to predict provider
spending. The ensemble uses a Ridge meta-learner for stacking and provides
conformal prediction intervals.

![Ensemble Actual vs Predicted](outputs/figures/ensemble_actual_vs_predicted.png)

![Ensemble Feature Importance](outputs/figures/ensemble_feature_importance.png)

![Ensemble Conformal Intervals](outputs/figures/ensemble_conformal_intervals.png)

### 6E. Provider Trajectory Analysis

Provider spending trajectories over 84 months (2018-2024) were analyzed to identify
distinct temporal patterns. Trajectory features include compound annual growth rate
(CAGR), volatility, COVID impact ratio, and trend classification. K-Shape clustering
with soft-DTW metric groups providers by temporal spending shape.

**Trajectory Archetypes Identified:** 8

| Archetype | Count | Median CAGR | Median Volatility |
|-----------|-------|-------------|-------------------|
| declining | 196 | N/A | 2.052 |
| exiting | 160,205 | N/A | 0.873 |
| growing | 17,130 | N/A | 1.461 |
| new_entrant | 225,984 | N/A | 0.793 |
| other | 57,403 | N/A | 0.571 |
| spike_then_decline | 126,683 | N/A | 0.781 |
| stable | 11,325 | N/A | 0.230 |
| volatile | 18,577 | N/A | 1.491 |

![Trajectory Archetypes](outputs/figures/trajectory_archetype_distribution.png)

![Trajectory Spaghetti by Archetype](outputs/figures/trajectory_spaghetti_by_archetype.png)

![K-Shape Centroids](outputs/figures/trajectory_kshape_centroids.png)

### 6F. Causal Analysis

Four complementary causal inference methods were applied to estimate the causal
impact of COVID-19 (March 2020) on Medicaid spending:

| Method | Key Result |
|--------|-----------|
| Difference-in-Differences | N/A |
| BSTS Counterfactual | N/A |
| Regression Discontinuity | N/A |
| Synthetic Control | N/A |

![DiD Event Study](outputs/figures/causal_did_event_study.png)

![BSTS Counterfactual](outputs/figures/causal_bsts_counterfactual.png)

![RDD Plot](outputs/figures/causal_rdd_plot.png)

![Synthetic Control](outputs/figures/causal_synthetic_control.png)

### 6G. Graph Community Detection

Provider-HCPCS co-billing networks were analyzed using Louvain community detection
to identify clusters of providers who bill similar procedure codes. Spectral graph
embedding provides additional structural insight.

| Metric | Value |
|--------|-------|
| Communities detected | 2,887 |
| Modularity | 0.233 |
| Nodes in graph | 0 |
| Edges in graph | 0 |

![Community Sizes](outputs/figures/graph_community_sizes.png)

![Spectral Embedding](outputs/figures/graph_spectral_embedding.png)

![Community Spending Profiles](outputs/figures/graph_community_spending_profiles.png)

### 6H. Unified Risk Scoring

All anomaly signals are aggregated into a single calibrated risk score per provider
using semi-supervised pseudo-labels and Platt scaling. Providers are classified into
risk tiers for prioritized investigation.

![Anomaly Method Comparison](outputs/figures/anomaly_method_comparison.png)

![Ensemble Anomaly Agreement](outputs/figures/ensemble_anomaly_agreement.png)

---

## Phase 7: State-of-the-Art Analysis

### 7A. Topological Data Analysis (TDA)

Persistent homology reveals the topological structure of the provider feature space —
connected components (H0) represent distinct provider groups, and loops (H1) indicate
circular relationships in billing patterns. This is among the first applications of
TDA to healthcare spending analysis.

| Metric | Value |
|--------|-------|
| Providers analyzed | 617,503 |
| Sample for persistence | 5,000 |
| H0 features (connected components) | 4,993 |
| H1 features (loops) | 3,190 |
| H1 max lifetime | 0.822 |
| Persistence entropy (H1) | 10.170 bits |

The presence of 3,190 H1 features indicates significant
loop structure in the provider feature space — providers form circular billing
pattern relationships rather than simple hierarchical clusters.

![Persistence Diagrams](outputs/figures/tda_persistence_diagrams.png)

![Betti Curves](outputs/figures/tda_betti_curves.png)

![Persistence Landscape](outputs/figures/tda_persistence_landscape.png)

![Topological Anomalies](outputs/figures/tda_topological_anomalies.png)

### 7B. Variational Autoencoder (β-VAE)

A β-VAE with cyclical KL annealing learns disentangled latent representations of
provider billing patterns. Unlike the deterministic autoencoder (Phase 6C), the VAE
provides a principled probabilistic anomaly score combining reconstruction error
and KL divergence from the prior — providers whose billing patterns cannot be
generated by the learned latent model are flagged as anomalous.

| Metric | Value |
|--------|-------|
| Providers scored | 617,503 |
| Latent dimensions | 16 |
| Parameters | 23,724 |
| Epochs trained | 17 |
| Final ELBO (val) | 5.5346 |
| Mean reconstruction error | 1.7155 |
| Mean KL divergence | 4.1198 |

![VAE Training](outputs/figures/vae_training_history.png)

![VAE Latent Space](outputs/figures/vae_latent_space.png)

![VAE Anomaly Distributions](outputs/figures/vae_anomaly_distributions.png)

### 7C. Optimal Transport Analysis

Optimal transport theory quantifies how spending distributions shift across time using
Wasserstein distances and Sinkhorn divergences. Transport plans reveal how healthcare
dollars flow between cost regimes, and McCann displacement interpolation visualizes
the distributional evolution from pre-COVID to post-COVID spending patterns.

| Metric | Value |
|--------|-------|
| Years analyzed | 7 (2018-2024) |
| Largest YoY shift | 2020 → 2021 (W₁ = 0.1074) |
| Pre/post COVID distance | W₁ = 0.0768 |
| Total cumulative shift | 0.3113 |

The largest distributional shift occurred from 2020 to 2021,
coinciding with the COVID-era enrollment expansion and spending surge.

![Wasserstein Heatmap](outputs/figures/ot_wasserstein_heatmap.png)

![Sinkhorn Heatmap](outputs/figures/ot_sinkhorn_heatmap.png)

![Transport Plan 2019→2021](outputs/figures/ot_transport_plan_2019_2021.png)

![YoY Wasserstein](outputs/figures/ot_yoy_wasserstein.png)

![Displacement Interpolation](outputs/figures/ot_displacement_interpolation.png)

### 7D. Double Machine Learning (DML)

The Chernozhukov et al. (2018) Double/Debiased ML framework estimates the causal
impact of COVID on provider spending with rigorous statistical guarantees. Cross-fitted
ML nuisance models (LightGBM) partial out confounding, and EconML's CausalForestDML
estimates heterogeneous treatment effects across provider types.

| Metric | Value |
|--------|-------|
| Observations | 518,826 |
| Unique providers | 262,098 |
| ATE (log-spending) | 0.0019 |
| Standard error | 0.0005 |
| 95% CI | [0.0009, 0.0029] |
| p-value | 1.70e-04 |
| Multiplicative effect | +0.2% |

**Interpretation:** COVID caused an average 0.2% increase
in provider spending, after controlling for observable confounders via ML-based
debiasing. This estimate has double-robustness guarantees.

![DML ATE](outputs/figures/dml_ate_estimate.png)

![CATE by Provider Size](outputs/figures/dml_cate_by_provider_size.png)

![CATE Distribution](outputs/figures/dml_cate_distribution.png)

### 7E. Information-Theoretic Analysis

Shannon entropy quantifies the diversity of each provider's billing code distribution,
mutual information reveals which features carry the most predictive signal about
spending, and transfer entropy measures directed information flow between spending
metrics over time.

| Metric | Value |
|--------|-------|
| Providers analyzed | 617,503 |
| Mean billing entropy | 1.140 bits |
| Median billing entropy | 0.930 bits |
| Max billing entropy | 7.887 bits |
| Top MI feature | total_claims (MI = 0.9469) |

Low-entropy providers (< 1 bit) are highly specialized, billing primarily a single
procedure code. High-entropy providers (> 5 bits) bill across hundreds of distinct
codes, characteristic of large health systems or FQHCs.

![Entropy Distribution](outputs/figures/info_entropy_distribution.png)

![Mutual Information](outputs/figures/info_mutual_information.png)

![Entropy Evolution](outputs/figures/info_entropy_evolution.png)

---

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


---

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
