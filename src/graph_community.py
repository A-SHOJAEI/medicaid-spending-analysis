"""Graph community detection, spectral analysis, and billing pattern communities.

Scales bipartite network to 20K providers, detects communities via Louvain,
computes spectral embeddings, and profiles community spending/anomaly patterns.
"""

import argparse
import gc
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency, format_number,
)

logger = setup_logging()


def build_extended_bipartite_graph(
    config: Optional[dict] = None,
    max_providers: int = 20000,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """Build bipartite provider-HCPCS graph for top providers."""
    if config is None:
        config = load_config()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Top providers by spending
    pf = pd.read_parquet(processed_dir / "provider_features.parquet",
                         columns=["billing_npi", "total_paid"])
    top_providers = set(pf.nlargest(max_providers, "total_paid")["billing_npi"].values)
    provider_list = sorted(top_providers)
    logger.info(f"Selected {len(provider_list):,} top providers")
    del pf

    # Build edges from all year files
    edges = Counter()
    for pf_path in sorted(processed_dir.glob("medicaid_*.parquet")):
        logger.info(f"Building graph from {pf_path.name}...")
        df = pd.read_parquet(pf_path, columns=[
            "BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE", "TOTAL_PAID"
        ])
        df = df[df["BILLING_PROVIDER_NPI_NUM"].isin(top_providers)]
        for (npi, hcpcs), total in df.groupby(
            ["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE"]
        )["TOTAL_PAID"].sum().items():
            edges[(npi, hcpcs)] += total
        del df
        gc.collect()

    # Build NetworkX graph
    G = nx.Graph()
    all_hcpcs = set()
    for (npi, hcpcs), weight in edges.items():
        G.add_node(f"P_{npi}", bipartite=0)
        G.add_node(f"H_{hcpcs}", bipartite=1)
        G.add_edge(f"P_{npi}", f"H_{hcpcs}", weight=weight)
        all_hcpcs.add(hcpcs)

    hcpcs_list = sorted(all_hcpcs)
    logger.info(f"Bipartite graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    logger.info(f"  Providers: {len(provider_list):,}, HCPCS: {len(hcpcs_list):,}")

    return G, np.array(provider_list), np.array(hcpcs_list)


def project_to_unipartite(
    G_bipartite: nx.Graph,
    provider_list: np.ndarray,
    hcpcs_list: np.ndarray,
    min_shared_codes: int = 3,
) -> Tuple[nx.Graph, sparse.csr_matrix]:
    """Project bipartite graph to unipartite provider graph."""
    n_prov = len(provider_list)
    n_hcpcs = len(hcpcs_list)

    prov_to_idx = {p: i for i, p in enumerate(provider_list)}
    hcpcs_to_idx = {h: i for i, h in enumerate(hcpcs_list)}

    # Build biadjacency matrix (binary: provider bills code or not)
    rows, cols = [], []
    for u, v in G_bipartite.edges():
        if u.startswith("P_") and v.startswith("H_"):
            npi = u[2:]
            hcpcs = v[2:]
        elif v.startswith("P_") and u.startswith("H_"):
            npi = v[2:]
            hcpcs = u[2:]
        else:
            continue
        if npi in prov_to_idx and hcpcs in hcpcs_to_idx:
            rows.append(prov_to_idx[npi])
            cols.append(hcpcs_to_idx[hcpcs])

    B = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_prov, n_hcpcs),
    )

    # Provider co-occurrence: P = B @ B.T
    logger.info("Computing provider co-occurrence matrix...")
    P = (B @ B.T).tocsr()

    # Build unipartite graph
    G_prov = nx.Graph()
    for i in range(n_prov):
        G_prov.add_node(provider_list[i])

    cx = P.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j and v >= min_shared_codes:
            G_prov.add_edge(provider_list[i], provider_list[j], weight=float(v))

    logger.info(f"Unipartite graph: {G_prov.number_of_nodes()} nodes, "
                f"{G_prov.number_of_edges()} edges (min_shared={min_shared_codes})")

    return G_prov, B


def louvain_community_detection(
    G_provider: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42,
) -> Tuple[dict, float]:
    """Detect communities using the Louvain algorithm."""
    import community as community_louvain

    partition = community_louvain.best_partition(
        G_provider, resolution=resolution, random_state=random_state
    )
    modularity = community_louvain.modularity(partition, G_provider)

    n_communities = len(set(partition.values()))
    logger.info(f"Louvain: {n_communities} communities, modularity={modularity:.4f}")

    return partition, modularity


def spectral_embedding(
    G_provider: nx.Graph,
    n_components: int = 10,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Compute spectral embedding of the provider adjacency matrix."""
    # Get largest connected component
    components = list(nx.connected_components(G_provider))
    largest = max(components, key=len)
    G_sub = G_provider.subgraph(largest)
    nodes_list = sorted(G_sub.nodes())
    n = len(nodes_list)

    logger.info(f"Spectral embedding on largest component: {n} nodes")

    # Build adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    rows, cols, data = [], [], []
    for u, v, d in G_sub.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = d.get("weight", 1.0)
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # Normalized Laplacian
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
    L_norm = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Compute smallest eigenvalues (skip trivial zero eigenvalue)
    from scipy.sparse.linalg import eigsh
    k = min(n_components + 1, n - 1)
    eigenvalues, eigenvectors = eigsh(L_norm, k=k, which="SM")

    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Skip the first (trivial) eigenvector
    embedding = eigenvectors[:, 1:n_components + 1]

    logger.info(f"Spectral embedding: {embedding.shape}, "
                f"eigenvalues: {eigenvalues[:5].tolist()}")

    return embedding, eigenvalues, nodes_list


def community_spending_profiles(
    partition: dict,
    providers_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute spending and anomaly profiles for each community."""
    # Build mapping
    comm_df = pd.DataFrame({
        "billing_npi": list(partition.keys()),
        "community": list(partition.values()),
    })

    # Merge with provider features
    merged = comm_df.merge(providers_df, on="billing_npi", how="left")

    # Merge anomaly flags
    if "fraud_flags_count" in anomaly_df.columns:
        anomaly_cols = ["billing_npi", "fraud_flags_count", "composite_anomaly_score"]
        available = [c for c in anomaly_cols if c in anomaly_df.columns]
        merged = merged.merge(anomaly_df[available], on="billing_npi", how="left")

    # Profile per community
    profiles = merged.groupby("community").agg(
        size=("billing_npi", "count"),
        total_spending=("total_paid", "sum"),
        median_spending=("total_paid", "median"),
        mean_claims=("total_claims", "mean"),
        median_hcpcs=("n_unique_hcpcs", "median"),
        median_servicing=("n_servicing_npis", "median"),
        anomaly_rate=("fraud_flags_count", lambda x: (x >= 2).mean() if "fraud_flags_count" in merged.columns else np.nan),
    ).reset_index()

    profiles = profiles.sort_values("total_spending", ascending=False)
    return profiles


def plot_graph_community(
    G_provider: nx.Graph,
    partition: dict,
    spectral_emb: np.ndarray,
    eigenvalues: np.ndarray,
    nodes_list: list,
    community_profiles: pd.DataFrame,
    config: dict,
) -> None:
    """Generate graph community visualizations."""
    setup_plotting(config)

    # 1. Spectral embedding scatter colored by community
    fig, ax = plt.subplots(figsize=(12, 10))
    node_communities = [partition.get(n, -1) for n in nodes_list]
    n_comm = len(set(node_communities))
    cmap = plt.cm.get_cmap("tab20", max(n_comm, 2))
    sc = ax.scatter(spectral_emb[:, 0], spectral_emb[:, 1],
                    c=node_communities, cmap=cmap, s=3, alpha=0.5, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Community ID")
    ax.set_xlabel("Spectral Component 1")
    ax.set_ylabel("Spectral Component 2")
    ax.set_title(f"Spectral Embedding — {n_comm} Louvain Communities")
    save_figure(fig, "graph_spectral_embedding", config)
    plt.close(fig)

    # 2. Community size distribution
    comm_sizes = pd.Series(list(partition.values())).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(comm_sizes.index, comm_sizes.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Provider Count")
    ax.set_title(f"Community Size Distribution ({len(comm_sizes)} communities)")
    ax.set_yscale("log")
    save_figure(fig, "graph_community_sizes", config)
    plt.close(fig)

    # 3. Eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(eigenvalues)), eigenvalues, "bo-", markersize=5)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Laplacian Eigenvalue Spectrum (Spectral Gap Analysis)")
    ax.grid(True, alpha=0.3)
    # Mark spectral gap
    if len(eigenvalues) > 2:
        gaps = np.diff(eigenvalues)
        max_gap_idx = np.argmax(gaps[1:]) + 1  # skip first
        ax.axvline(max_gap_idx, color="red", linestyle="--", alpha=0.5,
                   label=f"Largest spectral gap at k={max_gap_idx}")
        ax.legend()
    save_figure(fig, "graph_eigenvalue_spectrum", config)
    plt.close(fig)

    # 4. Community anomaly rates
    if "anomaly_rate" in community_profiles.columns:
        top_comm = community_profiles.nlargest(20, "size")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh([f"C{c} (n={s:,})" for c, s in zip(top_comm["community"], top_comm["size"])],
                top_comm["anomaly_rate"] * 100, color="orangered", edgecolor="white")
        ax.set_xlabel("Anomaly Rate (%)")
        ax.set_title("Anomaly Rate by Community (2+ fraud flags)")
        save_figure(fig, "graph_community_anomaly_rates", config)
        plt.close(fig)

    # 5. Community spending profiles
    top_comm = community_profiles.nlargest(10, "total_spending")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh([f"C{c}" for c in top_comm["community"]],
                 top_comm["total_spending"] / 1e9, color="steelblue")
    axes[0].set_xlabel("Total Spending ($B)")
    axes[0].set_title("Top 10 Communities by Spending")

    axes[1].barh([f"C{c}" for c in top_comm["community"]],
                 top_comm["median_hcpcs"], color="teal")
    axes[1].set_xlabel("Median HCPCS Codes per Provider")
    axes[1].set_title("Procedure Diversity by Community")
    plt.tight_layout()
    save_figure(fig, "graph_community_spending_profiles", config)
    plt.close(fig)


def run_graph_community(config: Optional[dict] = None) -> dict:
    """Run the full graph community analysis pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    gc_cfg = config.get("graph_community", {})
    seed = config["analysis"]["random_seed"]

    max_providers = gc_cfg.get("max_providers", 20000)
    min_shared = gc_cfg.get("min_shared_codes", 3)

    # Step 1: Build bipartite graph
    logger.info(f"Building bipartite graph for top {max_providers:,} providers...")
    G_bipartite, provider_list, hcpcs_list = build_extended_bipartite_graph(
        config, max_providers=max_providers
    )

    # Step 2: Project to unipartite
    logger.info("Projecting to unipartite provider graph...")
    G_prov, B = project_to_unipartite(
        G_bipartite, provider_list, hcpcs_list, min_shared_codes=min_shared
    )

    # Step 3: Louvain community detection
    logger.info("Running Louvain community detection...")
    partition, modularity = louvain_community_detection(
        G_prov, resolution=gc_cfg.get("louvain_resolution", 1.0), random_state=seed
    )

    # Step 4: Spectral embedding
    logger.info("Computing spectral embedding...")
    n_spectral = gc_cfg.get("spectral_n_components", 10)
    spectral_emb, eigenvalues, nodes_list = spectral_embedding(G_prov, n_components=n_spectral)

    # Step 5: Community profiles
    logger.info("Computing community profiles...")
    providers_df = pd.read_parquet(processed_dir / "provider_features.parquet")
    anomaly_df = pd.read_parquet(processed_dir / "provider_anomaly_scores.parquet",
                                  columns=["billing_npi", "fraud_flags_count", "composite_anomaly_score"])
    profiles = community_spending_profiles(partition, providers_df, anomaly_df)
    save_table(profiles, "graph_community_profiles", config)

    # Save community assignments
    comm_df = pd.DataFrame({
        "billing_npi": list(partition.keys()),
        "community": list(partition.values()),
    })
    # Add spectral coordinates for nodes in largest component
    node_to_spectral = {n: spectral_emb[i] for i, n in enumerate(nodes_list)}
    for d in range(min(n_spectral, spectral_emb.shape[1])):
        comm_df[f"spectral_{d}"] = comm_df["billing_npi"].map(
            lambda npi, dim=d: node_to_spectral.get(npi, [np.nan] * (dim + 1))[dim]
            if npi in node_to_spectral else np.nan
        )
    comm_df.to_parquet(processed_dir / "provider_communities.parquet", index=False)

    # Extended network metrics
    metrics = {
        "n_providers": int(len(provider_list)),
        "n_hcpcs": int(len(hcpcs_list)),
        "bipartite_nodes": int(G_bipartite.number_of_nodes()),
        "bipartite_edges": int(G_bipartite.number_of_edges()),
        "unipartite_nodes": int(G_prov.number_of_nodes()),
        "unipartite_edges": int(G_prov.number_of_edges()),
        "n_communities": int(len(set(partition.values()))),
        "modularity": float(modularity),
        "n_connected_components": int(nx.number_connected_components(G_prov)),
    }
    save_table(pd.DataFrame([metrics]), "graph_metrics_extended", config)

    # Step 6: Plots
    logger.info("Generating graph community plots...")
    plot_graph_community(G_prov, partition, spectral_emb, eigenvalues,
                         nodes_list, profiles, config)

    logger.info("Graph community analysis complete")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run graph community detection")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_graph_community(config)
