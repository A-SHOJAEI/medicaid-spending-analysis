"""Phase 7D: Double Machine Learning for heterogeneous causal effect estimation.

Implements the Chernozhukov et al. (2018) Double/Debiased ML framework to
estimate the causal impact of COVID on provider spending with heterogeneous
treatment effects across provider types.

Methods:
    - DML with cross-fitted LightGBM nuisance models
    - Conditional Average Treatment Effect (CATE) estimation
    - Treatment effect heterogeneity by provider size and specialization
    - Causal Forest for fine-grained CATE
    - Policy-relevant subgroup analysis
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Data Preparation ──────────────────────────────────────────────────


def prepare_dml_data(config: dict) -> pd.DataFrame:
    """Build a pre/post COVID panel for DML estimation."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load provider features
    feat_df = pd.read_parquet(processed_dir / "provider_features.parquet")
    logger.info(f"Loaded {len(feat_df)} providers")

    # Compute pre-COVID (2018-2019) and post-COVID (2021-2022) spending
    pre_years, post_years = [2018, 2019], [2021, 2022]
    pre_spending = {}
    post_spending = {}

    for year in pre_years + post_years:
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            logger.warning(f"Missing {path}")
            continue
        yr_df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "TOTAL_PAID",
                                                 "TOTAL_CLAIMS", "TOTAL_UNIQUE_BENEFICIARIES"])
        agg = yr_df.groupby("BILLING_PROVIDER_NPI_NUM").agg(
            total_paid=("TOTAL_PAID", "sum"),
            total_claims=("TOTAL_CLAIMS", "sum"),
            total_bene=("TOTAL_UNIQUE_BENEFICIARIES", "sum"),
        ).reset_index()

        if year in pre_years:
            pre_spending[year] = agg
        else:
            post_spending[year] = agg
        del yr_df
        gc.collect()

    # Average pre and post spending
    if not pre_spending or not post_spending:
        logger.error("Insufficient data for DML analysis")
        return pd.DataFrame()

    pre_all = pd.concat(pre_spending.values())
    pre_agg = pre_all.groupby("BILLING_PROVIDER_NPI_NUM").agg(
        pre_paid=("total_paid", "mean"),
        pre_claims=("total_claims", "mean"),
        pre_bene=("total_bene", "mean"),
    ).reset_index()

    post_all = pd.concat(post_spending.values())
    post_agg = post_all.groupby("BILLING_PROVIDER_NPI_NUM").agg(
        post_paid=("total_paid", "mean"),
        post_claims=("total_claims", "mean"),
        post_bene=("total_bene", "mean"),
    ).reset_index()

    # Merge: only providers active in both periods
    panel = pre_agg.merge(post_agg, on="BILLING_PROVIDER_NPI_NUM")
    panel = panel.merge(
        feat_df[["billing_npi", "n_unique_hcpcs", "n_servicing_npis",
                 "n_years_active", "n_months_active"]],
        left_on="BILLING_PROVIDER_NPI_NUM", right_on="billing_npi", how="inner"
    )

    # Treatment: post-COVID indicator (all rows are observed in both periods)
    # For DML, reshape to long format: each provider appears twice
    pre_rows = panel[["BILLING_PROVIDER_NPI_NUM", "pre_paid", "pre_claims", "pre_bene",
                       "n_unique_hcpcs", "n_servicing_npis", "n_years_active"]].copy()
    pre_rows.columns = ["npi", "spending", "claims", "beneficiaries",
                         "n_hcpcs", "n_servicing", "n_years"]
    pre_rows["treatment"] = 0

    post_rows = panel[["BILLING_PROVIDER_NPI_NUM", "post_paid", "post_claims", "post_bene",
                        "n_unique_hcpcs", "n_servicing_npis", "n_years_active"]].copy()
    post_rows.columns = ["npi", "spending", "claims", "beneficiaries",
                          "n_hcpcs", "n_servicing", "n_years"]
    post_rows["treatment"] = 1

    dml_df = pd.concat([pre_rows, post_rows], ignore_index=True)

    # Filter positive spending and take log
    dml_df = dml_df[dml_df["spending"] > 0].copy()
    dml_df["log_spending"] = np.log(dml_df["spending"])

    # Covariates
    dml_df["log_claims"] = np.log(dml_df["claims"].clip(lower=1))
    dml_df["log_bene"] = np.log(dml_df["beneficiaries"].clip(lower=1))
    dml_df["spending_per_claim"] = dml_df["spending"] / dml_df["claims"].clip(lower=1)
    dml_df["log_spending_per_claim"] = np.log(dml_df["spending_per_claim"].clip(lower=1))

    # Provider size categories (for heterogeneity analysis)
    dml_df["provider_size"] = pd.qcut(
        dml_df.groupby("npi")["spending"].transform("mean"),
        q=5, labels=["Very Small", "Small", "Medium", "Large", "Very Large"],
        duplicates="drop",
    )

    logger.info(f"DML panel: {dml_df.shape}, treatment split: "
                f"{(dml_df['treatment']==0).sum()} pre, {(dml_df['treatment']==1).sum()} post")
    return dml_df


# ── DML Estimation ────────────────────────────────────────────────────


def estimate_ate_dml(Y: np.ndarray, T: np.ndarray, X: np.ndarray,
                      n_folds: int = 5, seed: int = 42) -> dict:
    """Estimate ATE using Partially Linear DML with cross-fitting.

    Y = θ₀ T + g₀(X) + ε
    T = m₀(X) + V
    """
    from lightgbm import LGBMRegressor

    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Storage for residuals
    Y_res = np.zeros(n)
    T_res = np.zeros(n)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Nuisance model for E[Y|X]
        model_y = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, n_jobs=-1, verbose=-1,
                                 random_state=seed + fold)
        model_y.fit(X[train_idx], Y[train_idx])
        Y_res[test_idx] = Y[test_idx] - model_y.predict(X[test_idx])

        # Nuisance model for E[T|X]
        model_t = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, n_jobs=-1, verbose=-1,
                                 random_state=seed + fold + 100)
        model_t.fit(X[train_idx], T[train_idx])
        T_res[test_idx] = T[test_idx] - model_t.predict(X[test_idx])

    # DML estimator: θ̂ = Σ(T̃ᵢ Ỹᵢ) / Σ(T̃ᵢ²)
    theta = np.sum(T_res * Y_res) / np.sum(T_res ** 2)

    # Standard error via sandwich formula
    psi = T_res * (Y_res - theta * T_res)
    J = np.mean(T_res ** 2)
    se = np.sqrt(np.mean(psi ** 2)) / (J * np.sqrt(n))

    ci_low = theta - 1.96 * se
    ci_high = theta + 1.96 * se
    z_stat = theta / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))

    return {
        "ate": float(theta),
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "n_obs": n,
        "n_folds": n_folds,
    }


def estimate_cate_by_group(dml_df: pd.DataFrame, group_col: str,
                             covariate_cols: list, seed: int = 42) -> pd.DataFrame:
    """Estimate CATE for each subgroup."""
    results = []
    for group_val in sorted(dml_df[group_col].unique()):
        sub = dml_df[dml_df[group_col] == group_val]
        if len(sub) < 200:
            continue
        Y = sub["log_spending"].values
        T = sub["treatment"].values
        X = sub[covariate_cols].values

        try:
            res = estimate_ate_dml(Y, T, X, n_folds=3, seed=seed)
            res["group"] = str(group_val)
            res["n_group"] = len(sub)
            results.append(res)
        except Exception as e:
            logger.warning(f"  CATE estimation failed for {group_col}={group_val}: {e}")

    return pd.DataFrame(results)


def estimate_cate_econml(Y: np.ndarray, T: np.ndarray, X: np.ndarray,
                          seed: int = 42) -> dict:
    """Use EconML CausalForest for fine-grained CATE estimation."""
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        logger.warning("EconML not available, skipping CausalForest")
        return {}

    logger.info("Fitting CausalForestDML...")
    cf = CausalForestDML(
        model_y="auto",
        model_t="auto",
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    cf.fit(Y, T, X=X)

    # Individual treatment effects
    cate = cf.effect(X)
    cate_interval = cf.effect_interval(X, alpha=0.05)

    return {
        "model": cf,
        "cate": cate.ravel(),
        "cate_lower": cate_interval[0].ravel(),
        "cate_upper": cate_interval[1].ravel(),
    }


# ── Visualization ─────────────────────────────────────────────────────


def plot_ate_result(ate_result: dict, config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(8, 5))

    ate = ate_result["ate"]
    ci = [ate_result["ci_low"], ate_result["ci_high"]]

    ax.barh(["ATE (log-spending)"], [ate], xerr=[[ate - ci[0]], [ci[1] - ate]],
            color="#2196F3", alpha=0.8, capsize=10, height=0.4)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Treatment Effect (log-spending)")
    ax.set_title(f"DML Average Treatment Effect of COVID\n"
                 f"θ = {ate:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], "
                 f"p = {ate_result['p_value']:.2e})",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "dml_ate_estimate", config)


def plot_cate_by_group(cate_df: pd.DataFrame, group_col: str,
                        config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 6))

    groups = cate_df["group"].values
    ates = cate_df["ate"].values
    errors = np.array([ates - cate_df["ci_low"].values,
                       cate_df["ci_high"].values - ates])

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(groups)))
    ax.barh(groups, ates, xerr=errors, color=colors, alpha=0.8,
            capsize=8, height=0.5)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("CATE (log-spending)")
    ax.set_title(f"Heterogeneous Treatment Effects by {group_col}",
                 fontsize=13, fontweight="bold")

    for i, (g, a, p) in enumerate(zip(groups, ates, cate_df["p_value"].values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(max(a, 0) + 0.02, i, f"{a:.3f}{sig}", va="center", fontsize=9)

    save_figure(fig, f"dml_cate_by_{group_col}", config)


def plot_cate_distribution(cate_vals: np.ndarray, config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(cate_vals, bins=100, color="#9C27B0", alpha=0.7, density=True)
    axes[0].axvline(np.mean(cate_vals), color="red", ls="--", lw=2,
                    label=f"Mean = {np.mean(cate_vals):.4f}")
    axes[0].axvline(np.median(cate_vals), color="orange", ls="--", lw=2,
                    label=f"Median = {np.median(cate_vals):.4f}")
    axes[0].set_xlabel("Individual Treatment Effect (CATE)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Heterogeneous Treatment Effects")
    axes[0].legend()

    # QQ plot against normal
    sorted_cate = np.sort(cate_vals)
    theoretical = sp_stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_cate)))
    axes[1].scatter(theoretical, sorted_cate, alpha=0.3, s=2, color="#2196F3")
    axes[1].plot(theoretical, theoretical * np.std(cate_vals) + np.mean(cate_vals),
                 "r--", lw=1)
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("CATE Quantiles")
    axes[1].set_title("QQ Plot of CATE vs Normal")

    fig.suptitle("CausalForest CATE Analysis — COVID Impact Heterogeneity",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "dml_cate_distribution", config)


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_double_ml(config: dict) -> None:
    root = get_project_root()
    seed = config["analysis"]["random_seed"]

    # Prepare data
    logger.info("Preparing DML panel data...")
    dml_df = prepare_dml_data(config)
    if dml_df.empty:
        logger.error("No data for DML analysis")
        return

    covariate_cols = ["log_claims", "log_bene", "n_hcpcs", "n_servicing",
                       "n_years", "log_spending_per_claim"]

    Y = dml_df["log_spending"].values
    T = dml_df["treatment"].values
    X = dml_df[covariate_cols].fillna(0).values

    # ── ATE estimation ──
    logger.info("Estimating ATE via Double ML...")
    ate = estimate_ate_dml(Y, T, X, n_folds=5, seed=seed)
    logger.info(f"ATE = {ate['ate']:.4f} (SE={ate['se']:.4f}, p={ate['p_value']:.2e})")
    logger.info(f"  95% CI: [{ate['ci_low']:.4f}, {ate['ci_high']:.4f}]")
    logger.info(f"  Interpretation: COVID increased log-spending by {ate['ate']:.4f} "
                f"≈ {(np.exp(ate['ate'])-1)*100:.1f}% multiplicative effect")

    # ── CATE by provider size ──
    logger.info("Estimating CATE by provider size...")
    cate_size = estimate_cate_by_group(dml_df, "provider_size", covariate_cols, seed)
    if not cate_size.empty:
        save_table(cate_size, "dml_cate_by_provider_size", config)
        logger.info(f"CATE by size:\n{cate_size[['group', 'ate', 'se', 'p_value']].to_string()}")

    # ── CATE by HCPCS diversity ──
    dml_df["hcpcs_group"] = pd.qcut(
        dml_df["n_hcpcs"], q=4,
        labels=["Low diversity", "Medium-Low", "Medium-High", "High diversity"],
        duplicates="drop",
    )
    logger.info("Estimating CATE by HCPCS diversity...")
    cate_hcpcs = estimate_cate_by_group(dml_df, "hcpcs_group", covariate_cols, seed)
    if not cate_hcpcs.empty:
        save_table(cate_hcpcs, "dml_cate_by_hcpcs_diversity", config)

    # ── CausalForest CATE (subsample for speed) ──
    n_cf = min(100000, len(Y))
    rng = np.random.RandomState(seed)
    cf_idx = rng.choice(len(Y), n_cf, replace=False)
    logger.info(f"Running CausalForestDML on {n_cf} observations...")
    cf_result = estimate_cate_econml(Y[cf_idx], T[cf_idx], X[cf_idx], seed)

    if cf_result:
        cate_vals = cf_result["cate"]
        cate_df = pd.DataFrame({
            "npi": dml_df.iloc[cf_idx]["npi"].values,
            "cate": cate_vals,
            "cate_lower": cf_result["cate_lower"],
            "cate_upper": cf_result["cate_upper"],
        })
        cate_df.to_parquet(
            root / config["paths"]["processed_dir"] / "provider_dml_cate.parquet",
            index=False,
        )
        save_table(
            cate_df.describe().T, "dml_cate_summary_stats", config
        )
        plot_cate_distribution(cate_vals, config)
        logger.info(f"CausalForest CATE: mean={cate_vals.mean():.4f}, "
                    f"std={cate_vals.std():.4f}, "
                    f"range=[{cate_vals.min():.4f}, {cate_vals.max():.4f}]")

    # ── Visualizations ──
    plot_ate_result(ate, config)
    if not cate_size.empty:
        plot_cate_by_group(cate_size, "provider_size", config)
    if not cate_hcpcs.empty:
        plot_cate_by_group(cate_hcpcs, "hcpcs_group", config)

    # ── Summary ──
    summary = {
        "ate": ate,
        "ate_multiplicative_effect_pct": float((np.exp(ate["ate"]) - 1) * 100),
        "n_observations": int(len(dml_df)),
        "n_providers": int(dml_df["npi"].nunique()),
        "n_covariates": len(covariate_cols),
        "cate_mean": float(cf_result["cate"].mean()) if cf_result else None,
        "cate_std": float(cf_result["cate"].std()) if cf_result else None,
    }
    tables_dir = root / config["paths"]["tables_dir"]
    with open(tables_dir / "dml_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"DML summary: {json.dumps(summary, indent=2)}")

    gc.collect()
    logger.info("Phase 7D: Double Machine Learning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7D: Double Machine Learning")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_double_ml(config)
