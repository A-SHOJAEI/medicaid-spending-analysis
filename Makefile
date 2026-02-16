# Medicaid Provider Spending Analysis - Pipeline Orchestration
PYTHON = python3
SRC = src
CONFIG = config.yaml

.PHONY: all setup profile parquet features eda hypothesis anomaly clustering timeseries regression network geospatial embeddings advanced-clustering autoencoder ensemble trajectory causal graph-community risk-scoring tda vae optimal-transport double-ml info-theory report test clean

all: setup profile parquet features eda hypothesis anomaly clustering timeseries regression network geospatial embeddings advanced-clustering autoencoder ensemble trajectory causal graph-community risk-scoring tda vae optimal-transport double-ml info-theory report

setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/external outputs/figures outputs/tables outputs/models

# Phase 1: Data Profiling
profile:
	$(PYTHON) -m $(SRC).data_profiling --skip-parquet --config $(CONFIG)

parquet:
	$(PYTHON) -m $(SRC).data_profiling --skip-profile --config $(CONFIG)

# Phase 1b: Feature Engineering
features:
	$(PYTHON) -m $(SRC).feature_engineering --all --config $(CONFIG)

# Phase 2: Exploratory Data Analysis
eda:
	$(PYTHON) -m $(SRC).eda --config $(CONFIG)

# Phase 3: Hypothesis Testing
hypothesis:
	$(PYTHON) -m $(SRC).hypothesis_testing --config $(CONFIG)

# Phase 4A: Anomaly Detection
anomaly:
	$(PYTHON) -m $(SRC).anomaly_detection --config $(CONFIG)

# Phase 4B: Clustering
clustering:
	$(PYTHON) -m $(SRC).clustering --all --config $(CONFIG)

# Phase 4C: Time Series
timeseries:
	$(PYTHON) -m $(SRC).time_series --config $(CONFIG)

# Phase 4D: Regression / Predictive Modeling
regression:
	$(PYTHON) -m $(SRC).regression --config $(CONFIG)

# Phase 4E: Network Analysis
network:
	$(PYTHON) -m $(SRC).network_analysis --config $(CONFIG)

# Phase 4F: Geographic / Organizational Analysis
geospatial:
	$(PYTHON) -m $(SRC).geospatial --config $(CONFIG)

# Phase 6A: Provider Embeddings
embeddings:
	$(PYTHON) -m $(SRC).provider_embeddings --config $(CONFIG)

# Phase 6B: Advanced Clustering (UMAP + HDBSCAN)
advanced-clustering:
	$(PYTHON) -m $(SRC).advanced_clustering --config $(CONFIG)

# Phase 6C: Autoencoder Anomaly Detection
autoencoder:
	$(PYTHON) -m $(SRC).autoencoder_anomaly --config $(CONFIG)

# Phase 6D: Ensemble Gradient Boosting
ensemble:
	$(PYTHON) -m $(SRC).ensemble_model --config $(CONFIG)

# Phase 6E: Trajectory Analysis
trajectory:
	$(PYTHON) -m $(SRC).trajectory_analysis --config $(CONFIG)

# Phase 6F: Causal Analysis
causal:
	$(PYTHON) -m $(SRC).causal_analysis --config $(CONFIG)

# Phase 6G: Graph Community Detection
graph-community:
	$(PYTHON) -m $(SRC).graph_community --config $(CONFIG)

# Phase 6H: Risk Scoring
risk-scoring:
	$(PYTHON) -m $(SRC).risk_scoring --config $(CONFIG)

# Phase 7A: Topological Data Analysis
tda:
	$(PYTHON) -m $(SRC).topological_analysis --config $(CONFIG)

# Phase 7B: Variational Autoencoder
vae:
	$(PYTHON) -m $(SRC).vae_anomaly --config $(CONFIG)

# Phase 7C: Optimal Transport Analysis
optimal-transport:
	$(PYTHON) -m $(SRC).optimal_transport --config $(CONFIG)

# Phase 7D: Double Machine Learning
double-ml:
	$(PYTHON) -m $(SRC).double_ml --config $(CONFIG)

# Phase 7E: Information-Theoretic Analysis
info-theory:
	$(PYTHON) -m $(SRC).information_theory --config $(CONFIG)

# Phase 8: Generate Report
report:
	$(PYTHON) -m $(SRC).reporting

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v

# Clean generated outputs (preserves raw/processed data)
clean:
	rm -rf outputs/figures/*.png outputs/figures/*.pdf
	rm -rf outputs/tables/*.csv
	rm -rf outputs/models/*
