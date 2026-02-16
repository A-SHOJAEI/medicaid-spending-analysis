#!/bin/bash
# Medicaid Provider Spending Analysis - Full Pipeline
# Usage: bash run_all.sh
set -e

CONFIG="config.yaml"

echo "=== Medicaid Provider Spending Analysis Pipeline ==="
echo ""

# Phase 1: Data Profiling & Parquet Conversion
echo "[Phase 1] Data Profiling..."
python3 -m src.data_profiling --skip-parquet --config "$CONFIG"
echo "[Phase 1] Converting to Parquet..."
python3 -m src.data_profiling --skip-profile --config "$CONFIG"

# Phase 1b: Feature Engineering
echo "[Phase 1b] Building features..."
python3 -m src.feature_engineering --all --config "$CONFIG"

# Phase 2: Exploratory Data Analysis
echo "[Phase 2] Running EDA..."
python3 -m src.eda --config "$CONFIG"

# Phase 3: Hypothesis Testing
echo "[Phase 3] Running hypothesis tests..."
python3 -m src.hypothesis_testing --config "$CONFIG"

# Phase 4A: Anomaly Detection
echo "[Phase 4A] Running anomaly detection..."
python3 -m src.anomaly_detection --config "$CONFIG"

# Phase 4B: Clustering
echo "[Phase 4B] Running clustering..."
python3 -m src.clustering --all --config "$CONFIG"

# Phase 4C: Time Series
echo "[Phase 4C] Running time series analysis..."
python3 -m src.time_series --config "$CONFIG"

# Phase 4D: Regression
echo "[Phase 4D] Running predictive modeling..."
python3 -m src.regression --config "$CONFIG"

# Phase 4E: Network Analysis
echo "[Phase 4E] Running network analysis..."
python3 -m src.network_analysis --config "$CONFIG"

# Phase 4F: Geographic Analysis
echo "[Phase 4F] Running geographic analysis..."
python3 -m src.geospatial --config "$CONFIG"

# Phase 6A: Provider Embeddings
echo "[Phase 6A] Learning provider embeddings..."
python3 -m src.provider_embeddings --config "$CONFIG"

# Phase 6B: Advanced Clustering
echo "[Phase 6B] Running UMAP + HDBSCAN clustering..."
python3 -m src.advanced_clustering --config "$CONFIG"

# Phase 6C: Autoencoder Anomaly Detection
echo "[Phase 6C] Training autoencoder anomaly detector..."
python3 -m src.autoencoder_anomaly --config "$CONFIG"

# Phase 6D: Ensemble Gradient Boosting
echo "[Phase 6D] Training ensemble model..."
python3 -m src.ensemble_model --config "$CONFIG"

# Phase 6E: Trajectory Analysis
echo "[Phase 6E] Running trajectory analysis..."
python3 -m src.trajectory_analysis --config "$CONFIG"

# Phase 6F: Causal Analysis
echo "[Phase 6F] Running causal analysis..."
python3 -m src.causal_analysis --config "$CONFIG"

# Phase 6G: Graph Community Detection
echo "[Phase 6G] Running graph community detection..."
python3 -m src.graph_community --config "$CONFIG"

# Phase 6H: Risk Scoring
echo "[Phase 6H] Computing provider risk scores..."
python3 -m src.risk_scoring --config "$CONFIG"

# Phase 7A: Topological Data Analysis
echo "[Phase 7A] Running topological data analysis..."
python3 -m src.topological_analysis --config "$CONFIG"

# Phase 7B: Variational Autoencoder
echo "[Phase 7B] Training β-VAE anomaly detector..."
python3 -m src.vae_anomaly --config "$CONFIG"

# Phase 7C: Optimal Transport Analysis
echo "[Phase 7C] Running optimal transport analysis..."
python3 -m src.optimal_transport --config "$CONFIG"

# Phase 7D: Double Machine Learning
echo "[Phase 7D] Running double ML causal analysis..."
python3 -m src.double_ml --config "$CONFIG"

# Phase 7E: Information-Theoretic Analysis
echo "[Phase 7E] Running information-theoretic analysis..."
python3 -m src.information_theory --config "$CONFIG"

# Phase 8: Generate Report
echo "[Phase 8] Generating findings report..."
python3 -m src.reporting

echo ""
echo "=== Pipeline Complete ==="
echo "See FINDINGS_REPORT.md for results"
echo "See outputs/figures/ for visualizations"
echo "See outputs/tables/ for data tables"
