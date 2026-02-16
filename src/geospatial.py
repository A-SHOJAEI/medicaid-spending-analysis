"""State-level geographic analysis using NPI prefix patterns.

Note: The dataset does not contain a state column. Geographic analysis is limited
to what can be inferred from NPI numbers and aggregation patterns. This module
provides provider-count-based geographic proxies where feasible.
"""

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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table,
)

logger = setup_logging()


def analyze_provider_geographic_spread(config: Optional[dict] = None) -> pd.DataFrame:
    """Analyze geographic spread proxies based on provider billing patterns.

    Since we lack state data, we analyze the billing-servicing NPI relationship
    as a proxy for organizational spread.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with provider geographic spread metrics.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    providers = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet"
    )

    setup_plotting(config)

    # Distribution of servicing NPIs per billing NPI
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    svc_counts = providers["n_servicing_npis"].clip(upper=50)
    axes[0].hist(svc_counts, bins=50, color="#1976D2", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Number of Servicing NPIs")
    axes[0].set_ylabel("Number of Billing Providers")
    axes[0].set_title("Distribution of Servicing NPI Count per Billing Provider")
    axes[0].set_yscale("log")

    # Self-servicing vs delegated
    self_svc = providers["self_servicing"].sum()
    non_self = len(providers) - self_svc
    axes[1].bar(["Self-Servicing", "Uses Other NPIs"], [self_svc, non_self],
                color=["#4CAF50", "#FF7043"], alpha=0.8)
    axes[1].set_ylabel("Number of Providers")
    axes[1].set_title("Provider Self-Servicing vs Delegation")

    for i, v in enumerate([self_svc, non_self]):
        axes[1].text(i, v + 500, f"{v:,}\n({v/len(providers)*100:.1f}%)",
                     ha="center", fontsize=10)

    save_figure(fig, "provider_servicing_patterns", config)

    # Spending by organizational complexity
    bins = [0, 1, 2, 5, 10, 50, 1000]
    labels = ["1", "2", "3-5", "6-10", "11-50", "50+"]
    providers["svc_group"] = pd.cut(providers["n_servicing_npis"], bins=bins, labels=labels, right=True)

    group_stats = providers.groupby("svc_group", observed=True).agg(
        n_providers=("billing_npi", "count"),
        total_spending=("total_paid", "sum"),
        median_spending=("total_paid", "median"),
        mean_spending=("total_paid", "mean"),
    ).reset_index()

    save_table(group_stats, "spending_by_servicing_complexity", config)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(group_stats["svc_group"].astype(str), group_stats["total_spending"] / 1e9,
           color="#26A69A", alpha=0.85)
    ax.set_xlabel("Number of Servicing NPIs")
    ax.set_ylabel("Total Spending ($B)")
    ax.set_title("Total Spending by Organizational Complexity")
    save_figure(fig, "spending_by_complexity", config)

    logger.info("Geographic spread analysis complete")
    return group_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run geographic analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    analyze_provider_geographic_spread(config)
