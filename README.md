# Medicaid Provider Spending Analysis

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive data science analysis of the HHS/DOGE Medicaid Provider Spending dataset (T-MSIS), covering $1.09 trillion in provider-level Medicaid claims from 2018-2024 across 227 million records. Employs 30+ statistical and machine learning techniques across a 10-phase pipeline, including state-of-the-art methods from topological data analysis, optimal transport theory, and causal machine learning.

## Key Findings

- **Extreme provider concentration** (Gini = 0.925): 0.8% of providers account for 50% of all Medicaid spending; 469 mega-providers (0.08%) have median spending of $54.9M
- **59.4% post-COVID spending increase**: Medicaid spending surged from $108.7B (2018) to $198.8B (2023) before declining to $185.0B in 2024 during enrollment unwinding
- **Home care dominates costs**: HCPCS T1019 (Personal Care Services) alone is $122.7B (11.2% of total spending)
- **146,806 providers flagged** for anomalous billing patterns across 5 fraud signal categories; 6 providers triggered all 5 flags simultaneously
- **88 provider pairs** with >80% billing pattern similarity identified via network analysis
- **Causal impact of COVID**: BSTS model estimates $654B total causal impact (363% increase vs. counterfactual); DML confirms significant heterogeneous effects across provider types
- **2,887 provider communities** detected via Louvain graph clustering with modularity 0.233
- **3,190 topological loops (H1)** discovered in provider feature space via persistent homology
- **Optimal transport** reveals largest distributional shift between 2020→2021 (Wasserstein-1 = 0.107)

## Dataset

**Source:** [CMS T-MSIS Analytic Files](https://data.cms.gov/) — Medicaid Provider Spending
**Size:** 10.32 GB CSV, 227,083,361 rows, 7 columns
**Coverage:** Fee-for-service, managed care, and CHIP claims, January 2018 - December 2024
**Granularity:** Billing provider x Servicing provider x HCPCS code x Month
**Privacy:** Rows with <12 claims are suppressed

## Setup

```bash
# Clone the repository
git clone https://github.com/A-SHOJAEI/medicaid-spending-analysis.git
cd medicaid-spending-analysis

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Place the raw CSV file
# The config.yaml expects the CSV at ../medicaid-provider-spending.csv
# relative to the project root. Adjust paths in config.yaml if needed.
```

## Running the Full Pipeline

```bash
# Run everything end-to-end
make all

# Or run individual phases:
make profile             # Phase 1: Data profiling
make parquet             # Phase 1: Convert to Parquet
make features            # Phase 1b: Build feature vectors
make eda                 # Phase 2: Exploratory data analysis
make hypothesis          # Phase 3: Hypothesis testing
make anomaly             # Phase 4A: Anomaly detection
make clustering          # Phase 4B: Clustering
make timeseries          # Phase 4C: Time series analysis
make regression          # Phase 4D: Predictive modeling
make network             # Phase 4E: Network analysis
make embeddings          # Phase 6A: Provider embeddings (SVD + NMF)
make advanced-clustering # Phase 6B: UMAP + HDBSCAN clustering
make autoencoder         # Phase 6C: Autoencoder anomaly detection
make ensemble            # Phase 6D: Ensemble gradient boosting
make trajectory          # Phase 6E: Trajectory analysis
make causal              # Phase 6F: Causal analysis
make graph-community     # Phase 6G: Graph community detection
make risk-scoring        # Phase 6H: Unified risk scoring
make tda                 # Phase 7A: Topological data analysis
make vae                 # Phase 7B: β-VAE anomaly detection
make optimal-transport   # Phase 7C: Optimal transport analysis
make double-ml           # Phase 7D: Double machine learning
make info-theory         # Phase 7E: Information-theoretic analysis
make report              # Phase 8: Generate findings report

# Run tests
make test
```

## Repository Structure

```
medicaid-spending-analysis/
├── README.md                        # This file
├── FINDINGS_REPORT.md               # Comprehensive findings with statistical evidence
├── Makefile                         # Pipeline orchestration
├── requirements.txt                 # Python dependencies
├── config.yaml                      # Central configuration
│
├── data/
│   ├── raw/                         # Original CSV reference
│   ├── processed/                   # Year-partitioned Parquet files + feature vectors
│   └── external/                    # Supplementary reference data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Memory-efficient chunked CSV/Parquet loading
│   ├── data_cleaning.py             # Missing values, type casting, validation
│   ├── data_profiling.py            # Full dataset profiling and Parquet conversion
│   ├── feature_engineering.py       # Provider, HCPCS, and time series features
│   ├── eda.py                       # Exploratory visualizations
│   ├── hypothesis_testing.py        # 10 rigorous statistical hypothesis tests
│   ├── anomaly_detection.py         # Isolation Forest, LOF, fraud patterns, Benford
│   ├── clustering.py                # K-Means, GMM provider/HCPCS segmentation
│   ├── time_series.py               # STL, changepoints, SARIMA forecast, Granger
│   ├── regression.py                # LightGBM spending predictor with SHAP
│   ├── network_analysis.py          # Provider-HCPCS bipartite network analysis
│   ├── geospatial.py                # Provider organizational complexity analysis
│   ├── provider_embeddings.py       # SVD/NMF embeddings + k-NN anomaly scoring
│   ├── advanced_clustering.py       # UMAP + HDBSCAN density-based clustering
│   ├── autoencoder_anomaly.py       # Deep autoencoder anomaly detection
│   ├── ensemble_model.py            # LightGBM + XGBoost + CatBoost stacking with Optuna
│   ├── trajectory_analysis.py       # Provider spending trajectories + K-Shape clustering
│   ├── causal_analysis.py           # DiD, BSTS, RDD, Synthetic Control
│   ├── graph_community.py           # Louvain community detection on provider networks
│   ├── risk_scoring.py              # Unified calibrated risk scoring
│   ├── topological_analysis.py      # Persistent homology (TDA) + topological anomalies
│   ├── vae_anomaly.py               # β-VAE with cyclical KL annealing
│   ├── optimal_transport.py         # Wasserstein distances + transport plans
│   ├── double_ml.py                 # DML causal effects + CausalForest CATE
│   ├── information_theory.py        # Shannon entropy, MI, transfer entropy
│   ├── reporting.py                 # Auto-generates FINDINGS_REPORT.md
│   └── utils.py                     # Logging, plotting defaults, formatters
│
├── outputs/
│   ├── figures/                     # 80+ publication-quality plots (PNG + PDF)
│   ├── tables/                      # Summary statistics and test results (CSV/JSON)
│   └── models/                      # Serialized models (LightGBM, autoencoder, VAE, ensemble)
│
└── tests/
    └── test_data_integrity.py       # Data validation tests
```

## Methodology

### Data Processing
- 10.3 GB CSV processed in 2M-row chunks using pandas
- Converted to year-partitioned Snappy-compressed Parquet files
- Provider-level (617,503) and HCPCS-level (10,881) feature vectors

### Statistical Testing
- 10 hypotheses tested at alpha = 0.05
- Benjamini-Hochberg FDR correction for multiple comparisons
- Effect sizes and bootstrap confidence intervals reported
- 4/10 hypotheses significant after correction

### Anomaly Detection
- Ensemble: Modified Z-score + IQR + Isolation Forest + LOF
- Deep autoencoder reconstruction error
- β-VAE ELBO-based anomaly scoring (reconstruction + KL divergence)
- k-NN anomaly scoring on SVD/NMF embeddings
- Topological anomaly detection via persistent homology
- 5 rule-based fraud pattern signatures
- Composite scoring with ranked provider flagging
- Benford's Law first-digit analysis

### Machine Learning
- LightGBM regressor (R² = 0.832 on log-scale)
- Stacked ensemble: LightGBM + XGBoost + CatBoost with Optuna HPO (R² = 0.951) and conformal prediction intervals (90% coverage)
- K-Means clustering (silhouette = 0.993)
- UMAP + HDBSCAN density-based clustering
- SARIMA(1,1,1)(1,1,1,12) forecasting
- PELT changepoint detection

### Causal Inference
- Difference-in-Differences with event study
- Bayesian Structural Time Series (BSTS)
- Regression Discontinuity Design (RDD)
- Synthetic Control Method
- Double Machine Learning (Chernozhukov et al., 2018) for debiased ATE
- EconML CausalForestDML for heterogeneous treatment effects (CATE)

### Network & Graph Analysis
- Provider-HCPCS bipartite network analysis
- Louvain community detection (2,887 communities)
- Spectral graph embedding
- Provider trajectory analysis with K-Shape clustering

### Topological Data Analysis
- Vietoris-Rips persistent homology (H0, H1)
- Persistence diagrams, Betti curves, persistence landscapes
- Topological anomaly scoring via local neighborhood complexity

### Optimal Transport
- Wasserstein-1 and Wasserstein-2 distances between annual distributions
- Sinkhorn divergences with entropic regularization
- Transport plans and McCann displacement interpolation
- Year-over-year distributional shift quantification

### Information Theory
- Shannon entropy of provider billing code distributions
- Mutual information between features and spending outcomes
- Transfer entropy for directed temporal information flow
- Jensen-Shannon divergence for cluster comparison

### Risk Scoring
- Unified calibrated risk scores aggregating all anomaly signals
- Semi-supervised pseudo-labels with Platt scaling
- Tiered risk classification

## License

MIT License

## Acknowledgments

Data source: Centers for Medicare & Medicaid Services (CMS), T-MSIS Analytic Files.
This analysis is for research purposes only. Anomaly flags are statistical signals,
not determinations of fraud.
