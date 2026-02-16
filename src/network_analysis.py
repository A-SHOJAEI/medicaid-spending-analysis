"""Provider-procedure billing network analysis."""

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
import networkx as nx
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table,
)

logger = setup_logging()


def build_provider_hcpcs_network(config: Optional[dict] = None,
                                  max_providers: int = 5000) -> nx.Graph:
    """Build a bipartite graph of providers and HCPCS codes.

    Uses top providers by spending to keep the network manageable.

    Args:
        config: Configuration dictionary.
        max_providers: Maximum number of providers to include.

    Returns:
        NetworkX bipartite graph.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Get top providers
    providers = pd.read_parquet(
        processed_dir / "provider_features.parquet",
        columns=["billing_npi", "total_paid"],
    )
    top_npis = set(providers.nlargest(max_providers, "total_paid")["billing_npi"])

    # Build edges from parquet files
    edges = Counter()
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE", "TOTAL_PAID"])
        df = df[df["BILLING_PROVIDER_NPI_NUM"].isin(top_npis)]
        for (npi, hcpcs), total in df.groupby(["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE"])["TOTAL_PAID"].sum().items():
            edges[(f"P_{npi}", f"H_{hcpcs}")] += total
        del df

    # Build graph
    G = nx.Graph()
    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)

    # Tag node types
    for node in G.nodes():
        G.nodes[node]["bipartite"] = 0 if node.startswith("P_") else 1

    logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def analyze_network(G: nx.Graph, config: dict) -> dict:
    """Analyze network structure and centrality.

    Args:
        G: NetworkX graph.
        config: Configuration dictionary.

    Returns:
        Dictionary of network metrics.
    """
    setup_plotting(config)

    # Separate node types
    providers = [n for n in G.nodes() if n.startswith("P_")]
    hcpcs = [n for n in G.nodes() if n.startswith("H_")]

    # Degree distribution
    provider_degrees = [G.degree(n) for n in providers]
    hcpcs_degrees = [G.degree(n) for n in hcpcs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(provider_degrees, bins=50, color="#1565C0", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Degree (Number of HCPCS codes)")
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_title("Provider Degree Distribution")
    axes[0].set_yscale("log")

    axes[1].hist(hcpcs_degrees, bins=50, color="#E64A19", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Degree (Number of Providers)")
    axes[1].set_ylabel("Number of HCPCS Codes")
    axes[1].set_title("HCPCS Code Degree Distribution")
    axes[1].set_yscale("log")

    save_figure(fig, "network_degree_distribution", config)

    # HCPCS centrality (betweenness, on subset for performance)
    if G.number_of_nodes() < 10000:
        bc = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    else:
        bc = nx.degree_centrality(G)

    # Top hub HCPCS codes
    hcpcs_centrality = {n: bc.get(n, 0) for n in hcpcs}
    top_hcpcs = sorted(hcpcs_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(12, 7))
    codes = [h[0].replace("H_", "") for h in top_hcpcs]
    scores = [h[1] for h in top_hcpcs]
    ax.barh(range(len(codes)), scores, color="#26A69A", alpha=0.85)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes)
    ax.invert_yaxis()
    ax.set_xlabel("Centrality Score")
    ax.set_title("Top 20 HCPCS Codes by Network Centrality")
    save_figure(fig, "network_hcpcs_centrality", config)

    # Connected components
    n_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)

    metrics = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_providers": len(providers),
        "n_hcpcs": len(hcpcs),
        "n_connected_components": n_components,
        "largest_component_size": len(largest_cc),
        "avg_provider_degree": np.mean(provider_degrees),
        "avg_hcpcs_degree": np.mean(hcpcs_degrees),
        "density": nx.density(G),
        "top_hub_codes": [(c.replace("H_", ""), round(s, 4)) for c, s in top_hcpcs[:10]],
    }

    save_table(pd.DataFrame([metrics]), "network_metrics", config)
    logger.info(f"Network metrics: {metrics}")

    return metrics


def find_similar_billing_patterns(config: Optional[dict] = None,
                                   n_top: int = 2000) -> pd.DataFrame:
    """Find providers with suspiciously similar billing patterns.

    Uses Jaccard similarity on HCPCS code sets.

    Args:
        config: Configuration dictionary.
        n_top: Number of top providers to compare.

    Returns:
        DataFrame of provider pairs with high similarity.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Get HCPCS sets per provider for top providers
    providers = pd.read_parquet(
        processed_dir / "provider_features.parquet",
        columns=["billing_npi", "total_paid"],
    )
    top_npis = set(providers.nlargest(n_top, "total_paid")["billing_npi"])

    provider_codes = {}
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE"])
        df = df[df["BILLING_PROVIDER_NPI_NUM"].isin(top_npis)]
        for npi, group in df.groupby("BILLING_PROVIDER_NPI_NUM"):
            if npi not in provider_codes:
                provider_codes[npi] = set()
            provider_codes[npi].update(group["HCPCS_CODE"].dropna().unique())
        del df

    # Compute Jaccard similarities (for providers with many codes)
    npis = list(provider_codes.keys())
    similar_pairs = []

    for i in range(len(npis)):
        for j in range(i + 1, min(i + 200, len(npis))):  # limit comparisons
            set_i = provider_codes[npis[i]]
            set_j = provider_codes[npis[j]]
            if len(set_i) < 5 or len(set_j) < 5:
                continue
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 0
            if jaccard > 0.8:  # High similarity threshold
                similar_pairs.append({
                    "npi_1": npis[i],
                    "npi_2": npis[j],
                    "jaccard_similarity": jaccard,
                    "shared_codes": intersection,
                    "total_codes_union": union,
                })

    if similar_pairs:
        result = pd.DataFrame(similar_pairs).sort_values("jaccard_similarity", ascending=False)
        save_table(result, "similar_billing_patterns", config)
        logger.info(f"Found {len(result)} provider pairs with >80% billing similarity")
    else:
        result = pd.DataFrame()
        logger.info("No highly similar billing patterns found")

    return result


def run_network_analysis(config: Optional[dict] = None) -> None:
    """Run the full network analysis pipeline.

    Args:
        config: Configuration dictionary.
    """
    if config is None:
        config = load_config()

    G = build_provider_hcpcs_network(config, max_providers=3000)
    metrics = analyze_network(G, config)
    similar = find_similar_billing_patterns(config, n_top=1000)

    logger.info("Network analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run network analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_network_analysis(config)
