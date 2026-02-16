"""Anomaly and fraud signal detection in Medicaid provider spending."""

import argparse
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


def statistical_outliers(providers: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Flag statistical outliers using Z-score and IQR methods.

    Args:
        providers: Provider feature DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with outlier flags added.
    """
    providers = providers.copy()
    numeric_cols = ["total_paid", "total_claims", "paid_per_claim",
                    "n_unique_hcpcs", "n_servicing_npis"]

    for col in numeric_cols:
        if col not in providers.columns:
            continue
        values = providers[col].fillna(0)

        # Z-score (using robust median/MAD)
        median = values.median()
        mad = np.median(np.abs(values - median))
        mad = mad if mad > 0 else 1
        z_scores = 0.6745 * (values - median) / mad
        providers[f"{col}_zscore"] = z_scores
        providers[f"{col}_outlier_z"] = np.abs(z_scores) > 3

        # IQR method
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        providers[f"{col}_outlier_iqr"] = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)

    # Composite outlier score: count of flags
    outlier_cols = [c for c in providers.columns if c.endswith("_outlier_z") or c.endswith("_outlier_iqr")]
    providers["statistical_outlier_score"] = providers[outlier_cols].sum(axis=1)

    logger.info(f"Statistical outliers: {(providers['statistical_outlier_score'] > 0).sum():,} "
                f"providers flagged (out of {len(providers):,})")

    return providers


def benford_analysis(config: dict) -> pd.DataFrame:
    """Perform Benford's Law analysis on payment amounts.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with Benford test results.
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Collect first digits from all payments
    first_digits = []
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=["TOTAL_PAID"])
        paid = df["TOTAL_PAID"].dropna()
        paid = paid[paid > 0]
        digits = paid.apply(lambda x: int(str(f"{x:.2f}").lstrip("0").lstrip(".")[0])
                            if x > 0 else 0)
        digits = digits[digits > 0]
        first_digits.extend(digits.values)
        del df

    first_digits = np.array(first_digits)
    digit_counts = pd.Series(first_digits).value_counts().sort_index()
    total = len(first_digits)

    # Expected Benford distribution
    benford_expected = {d: np.log10(1 + 1 / d) for d in range(1, 10)}

    results = []
    for d in range(1, 10):
        observed_pct = digit_counts.get(d, 0) / total
        expected_pct = benford_expected[d]
        results.append({
            "digit": d,
            "observed_count": int(digit_counts.get(d, 0)),
            "observed_pct": observed_pct * 100,
            "expected_pct": expected_pct * 100,
            "deviation": (observed_pct - expected_pct) * 100,
        })

    benford_df = pd.DataFrame(results)

    # Chi-squared test
    observed = np.array([digit_counts.get(d, 0) for d in range(1, 10)])
    expected = np.array([benford_expected[d] * total for d in range(1, 10)])
    chi2, p_value = stats.chisquare(observed, f_exp=expected)

    # Plot
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, 10)
    width = 0.35
    ax.bar(x - width / 2, benford_df["observed_pct"], width, label="Observed", color="#1976D2", alpha=0.8)
    ax.bar(x + width / 2, benford_df["expected_pct"], width, label="Benford Expected", color="#F44336", alpha=0.8)
    ax.set_xlabel("First Digit")
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Benford's Law Analysis of Payment Amounts (chi2={chi2:.0f}, p={p_value:.2e})")
    ax.set_xticks(x)
    ax.legend()
    save_figure(fig, "benford_analysis", config)

    benford_df["chi2_statistic"] = chi2
    benford_df["chi2_p_value"] = p_value

    save_table(benford_df, "benford_analysis", config)
    logger.info(f"Benford analysis: chi2={chi2:.1f}, p={p_value:.2e}")

    return benford_df


def isolation_forest_detection(providers: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run Isolation Forest anomaly detection on provider features.

    Args:
        providers: Provider feature DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with anomaly scores.
    """
    feature_cols = ["total_paid", "total_claims", "paid_per_claim",
                    "paid_per_beneficiary", "n_unique_hcpcs",
                    "n_servicing_npis", "n_years_active", "n_months_active"]
    feature_cols = [c for c in feature_cols if c in providers.columns]

    X = providers[feature_cols].fillna(0).copy()
    # Log transform skewed features
    for col in ["total_paid", "total_claims", "paid_per_claim", "paid_per_beneficiary"]:
        if col in X.columns:
            X[col] = np.log1p(np.abs(X[col]))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = config["analysis"]["anomaly_contamination"]
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=config["analysis"]["random_seed"],
        n_jobs=-1,
    )
    providers = providers.copy()
    providers["isolation_forest_label"] = clf.fit_predict(X_scaled)
    providers["isolation_forest_score"] = clf.decision_function(X_scaled)

    n_anomalies = (providers["isolation_forest_label"] == -1).sum()
    logger.info(f"Isolation Forest: {n_anomalies:,} anomalies detected "
                f"({n_anomalies/len(providers)*100:.2f}%)")

    return providers


def local_outlier_factor(providers: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run Local Outlier Factor detection.

    Args:
        providers: Provider feature DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with LOF scores.
    """
    feature_cols = ["total_paid", "total_claims", "paid_per_claim",
                    "n_unique_hcpcs", "n_servicing_npis"]
    feature_cols = [c for c in feature_cols if c in providers.columns]

    X = providers[feature_cols].fillna(0).copy()
    for col in ["total_paid", "total_claims", "paid_per_claim"]:
        if col in X.columns:
            X[col] = np.log1p(np.abs(X[col]))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = config["analysis"]["anomaly_contamination"]
    clf = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1,
    )
    providers = providers.copy()
    providers["lof_label"] = clf.fit_predict(X_scaled)
    providers["lof_score"] = clf.negative_outlier_factor_

    n_anomalies = (providers["lof_label"] == -1).sum()
    logger.info(f"LOF: {n_anomalies:,} anomalies detected")

    return providers


def fraud_pattern_signatures(providers: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Detect specific fraud pattern signatures.

    Args:
        providers: Provider feature DataFrame with anomaly scores.
        config: Configuration dictionary.

    Returns:
        DataFrame with fraud pattern flags.
    """
    providers = providers.copy()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # 1. Excessive procedure code diversity (stuffing)
    hcpcs_95 = providers["n_unique_hcpcs"].quantile(0.95)
    providers["flag_code_stuffing"] = providers["n_unique_hcpcs"] > hcpcs_95

    # 2. Very high paid per claim
    ppc_99 = providers["paid_per_claim"].quantile(0.99)
    providers["flag_high_cost_per_claim"] = providers["paid_per_claim"] > ppc_99

    # 3. Many servicing NPIs (possible shell billing)
    providers["flag_many_servicing"] = providers["n_servicing_npis"] > 10

    # 4. Very high spending relative to peers
    spending_99 = providers["total_paid"].quantile(0.99)
    providers["flag_extreme_spending"] = providers["total_paid"] > spending_99

    # 5. Negative payment anomalies
    providers["flag_negative_payments"] = providers["has_negative_payments"]

    # 6. Detect year-over-year spending spikes (>300%)
    # This requires yearly data per provider
    yearly_spending = {}
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        year = int(pf.stem.split("_")[1])
        df = pd.read_parquet(pf, columns=["BILLING_PROVIDER_NPI_NUM", "TOTAL_PAID"])
        npi_spend = df.groupby("BILLING_PROVIDER_NPI_NUM")["TOTAL_PAID"].sum()
        for npi, spend in npi_spend.items():
            if npi not in yearly_spending:
                yearly_spending[npi] = {}
            yearly_spending[npi][year] = spend
        del df

    spike_npis = set()
    for npi, years_dict in yearly_spending.items():
        sorted_years = sorted(years_dict.keys())
        for i in range(1, len(sorted_years)):
            prev_year = sorted_years[i - 1]
            curr_year = sorted_years[i]
            prev_val = years_dict[prev_year]
            curr_val = years_dict[curr_year]
            if prev_val > 1000 and curr_val > prev_val * 4:  # 300% increase = 4x
                spike_npis.add(npi)
                break

    providers["flag_spending_spike"] = providers["billing_npi"].isin(spike_npis)

    # Composite fraud score
    flag_cols = [c for c in providers.columns if c.startswith("flag_")]
    providers["fraud_flags_count"] = providers[flag_cols].sum(axis=1)

    # Combined anomaly rank
    providers["composite_anomaly_score"] = 0.0
    if "isolation_forest_score" in providers.columns:
        providers["composite_anomaly_score"] -= providers["isolation_forest_score"]
    if "lof_score" in providers.columns:
        providers["composite_anomaly_score"] -= providers["lof_score"]
    if "statistical_outlier_score" in providers.columns:
        providers["composite_anomaly_score"] += providers["statistical_outlier_score"]
    providers["composite_anomaly_score"] += providers["fraud_flags_count"] * 2

    # Rank
    providers["anomaly_rank"] = providers["composite_anomaly_score"].rank(ascending=False, method="min")

    logger.info(f"Fraud pattern flags: {(providers['fraud_flags_count'] > 0).sum():,} providers with >= 1 flag")
    logger.info(f"  Code stuffing: {providers['flag_code_stuffing'].sum():,}")
    logger.info(f"  High cost/claim: {providers['flag_high_cost_per_claim'].sum():,}")
    logger.info(f"  Many servicing NPIs: {providers['flag_many_servicing'].sum():,}")
    logger.info(f"  Extreme spending: {providers['flag_extreme_spending'].sum():,}")
    logger.info(f"  Spending spikes: {providers['flag_spending_spike'].sum():,}")

    return providers


def plot_anomaly_results(providers: pd.DataFrame, config: dict) -> None:
    """Generate anomaly detection visualizations.

    Args:
        providers: Provider DataFrame with anomaly scores.
        config: Configuration dictionary.
    """
    setup_plotting(config)

    # 1. Anomaly score distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "isolation_forest_score" in providers.columns:
        axes[0].hist(providers["isolation_forest_score"], bins=100, color="#1565C0", alpha=0.8)
        axes[0].axvline(0, color="red", linestyle="--", label="Anomaly threshold")
        axes[0].set_xlabel("Isolation Forest Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Isolation Forest Anomaly Score Distribution")
        axes[0].legend()

    if "composite_anomaly_score" in providers.columns:
        axes[1].hist(providers["composite_anomaly_score"], bins=100, color="#E64A19", alpha=0.8)
        axes[1].set_xlabel("Composite Anomaly Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Composite Anomaly Score Distribution")
        axes[1].set_yscale("log")

    save_figure(fig, "anomaly_score_distribution", config)

    # 2. Scatter: spending vs claims colored by anomaly
    fig, ax = plt.subplots(figsize=(12, 8))
    normal = providers[providers.get("isolation_forest_label", 1) == 1]
    anomaly = providers[providers.get("isolation_forest_label", 1) == -1]

    ax.scatter(np.log10(normal["total_claims"].clip(lower=1)),
               np.log10(normal["total_paid"].clip(lower=1)),
               s=1, alpha=0.1, color="gray", label="Normal")
    ax.scatter(np.log10(anomaly["total_claims"].clip(lower=1)),
               np.log10(anomaly["total_paid"].clip(lower=1)),
               s=5, alpha=0.5, color="red", label="Anomaly")
    ax.set_xlabel("log10(Total Claims)")
    ax.set_ylabel("log10(Total Paid)")
    ax.set_title("Anomalous Providers: Spending vs Claims")
    ax.legend()
    save_figure(fig, "anomaly_scatter_spending_claims", config)

    # 3. Fraud flags breakdown
    flag_cols = [c for c in providers.columns if c.startswith("flag_")]
    if flag_cols:
        flag_counts = providers[flag_cols].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        flag_counts.plot(kind="barh", ax=ax, color="#FF7043", alpha=0.85)
        ax.set_xlabel("Number of Flagged Providers")
        ax.set_title("Fraud Pattern Signature Flags")
        labels = [c.replace("flag_", "").replace("_", " ").title() for c in flag_counts.index]
        ax.set_yticklabels(labels)
        save_figure(fig, "fraud_flags_breakdown", config)

    logger.info("Anomaly detection plots saved")


def run_anomaly_detection(config: Optional[dict] = None) -> pd.DataFrame:
    """Run the full anomaly detection pipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Provider DataFrame with all anomaly scores and flags.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    prov_path = root / config["paths"]["processed_dir"] / "provider_features.parquet"
    providers = pd.read_parquet(prov_path)

    logger.info(f"Running anomaly detection on {len(providers):,} providers")

    # Step 1: Statistical outliers
    providers = statistical_outliers(providers, config)

    # Step 2: Isolation Forest
    providers = isolation_forest_detection(providers, config)

    # Step 3: LOF
    providers = local_outlier_factor(providers, config)

    # Step 4: Fraud patterns
    providers = fraud_pattern_signatures(providers, config)

    # Step 5: Visualizations
    plot_anomaly_results(providers, config)

    # Save flagged providers table (top 100)
    top_anomalies = providers.nlargest(100, "composite_anomaly_score")
    display_cols = ["billing_npi", "total_paid", "total_claims", "paid_per_claim",
                    "n_unique_hcpcs", "n_servicing_npis", "n_years_active",
                    "composite_anomaly_score", "anomaly_rank", "fraud_flags_count",
                    "isolation_forest_label", "flag_code_stuffing",
                    "flag_high_cost_per_claim", "flag_many_servicing",
                    "flag_extreme_spending", "flag_spending_spike"]
    display_cols = [c for c in display_cols if c in top_anomalies.columns]
    save_table(top_anomalies[display_cols], "top_anomalous_providers", config)

    # Save full results
    output_path = root / config["paths"]["processed_dir"] / "provider_anomaly_scores.parquet"
    providers.to_parquet(output_path, index=False)
    logger.info(f"Anomaly detection complete. Results saved to {output_path}")

    # Benford's Law
    benford_analysis(config)

    return providers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_anomaly_detection(config)
