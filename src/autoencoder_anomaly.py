"""Deep autoencoder for anomaly detection on provider features.

Trains a PyTorch autoencoder on all 617K providers; reconstruction error
serves as a SOTA anomaly score. Combines with existing methods into an
ensemble anomaly ranking.
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


def get_device():
    """Detect best available device (CUDA > MPS > CPU)."""
    import torch
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU device")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        logger.info("Using Apple MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    logger.info("Using CPU device")
    return torch.device("cpu")


def prepare_features(providers: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """Prepare and log-transform provider features for the autoencoder."""
    feature_cols = [
        "total_paid", "total_claims", "total_beneficiaries",
        "paid_per_claim", "paid_per_beneficiary", "claims_per_beneficiary",
        "n_unique_hcpcs", "n_servicing_npis",
        "n_years_active", "n_months_active", "row_count",
        "neg_paid_ratio",
    ]
    # Ensure all columns exist
    available = [c for c in feature_cols if c in providers.columns]

    X = providers[available].copy()

    # Log-transform highly skewed columns
    log_cols = ["total_paid", "total_claims", "total_beneficiaries",
                "paid_per_claim", "paid_per_beneficiary", "row_count"]
    for col in log_cols:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    X = X.fillna(0).values.astype(np.float32)
    return X, available


class ProviderAutoencoder:
    """PyTorch autoencoder for provider feature reconstruction."""

    def __init__(self, input_dim: int, device=None):
        import torch
        import torch.nn as nn

        self.device = device or get_device()
        self.scaler = RobustScaler()
        self.input_dim = input_dim
        self.history = {"train_loss": [], "val_loss": []}

        # Build model
        class Autoencoder(nn.Module):
            def __init__(self, d_in):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(d_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(32, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, d_in),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

            def encode(self, x):
                return self.encoder(x)

        self.model = Autoencoder(input_dim).to(self.device)

    def fit(self, X: np.ndarray, batch_size=4096, epochs=100,
            validation_split=0.1, lr=1e-3, weight_decay=1e-4, patience=15):
        """Train the autoencoder."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Scale
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        # Train/val split
        n = len(X_scaled)
        n_val = int(n * validation_split)
        perm = np.random.RandomState(42).permutation(n)
        X_train = X_scaled[perm[n_val:]]
        X_val = X_scaled[perm[:n_val]]

        train_ds = TensorDataset(torch.from_numpy(X_train))
        val_ds = TensorDataset(torch.from_numpy(X_val))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = torch.nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = criterion(out, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_train)

            # Validate
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    out = self.model(batch)
                    loss = criterion(out, batch)
                    val_loss += loss.item() * len(batch)
            val_loss /= len(X_val)

            scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        self.model.load_state_dict(best_state)
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        return self.history

    def predict_reconstruction_error(self, X: np.ndarray, batch_size=8192) -> np.ndarray:
        """Compute per-sample reconstruction error."""
        import torch
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        errors = []

        with torch.no_grad():
            for start in range(0, len(X_scaled), batch_size):
                batch = torch.from_numpy(X_scaled[start:start + batch_size]).to(self.device)
                out = self.model(batch)
                mse = ((out - batch) ** 2).mean(dim=1).cpu().numpy()
                errors.append(mse)

        return np.concatenate(errors)

    def get_bottleneck_features(self, X: np.ndarray, batch_size=8192) -> np.ndarray:
        """Extract bottleneck (latent) representations."""
        import torch
        X_scaled = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        features = []

        with torch.no_grad():
            for start in range(0, len(X_scaled), batch_size):
                batch = torch.from_numpy(X_scaled[start:start + batch_size]).to(self.device)
                z = self.model.encode(batch).cpu().numpy()
                features.append(z)

        return np.concatenate(features)


def build_ensemble_scores(
    providers: pd.DataFrame,
    ae_scores: np.ndarray,
    embedding_anomaly: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Combine all anomaly scores into a rank-based ensemble."""
    df = providers[["billing_npi"]].copy()

    # Rank-normalize each score to [0, 1]
    def rank_normalize(arr):
        return stats.rankdata(arr) / len(arr)

    df["ae_rank"] = rank_normalize(ae_scores)

    if "isolation_forest_score" in providers.columns:
        df["if_rank"] = rank_normalize(-providers["isolation_forest_score"].values)
    if "lof_score" in providers.columns:
        df["lof_rank"] = rank_normalize(-providers["lof_score"].values)
    if "statistical_outlier_score" in providers.columns:
        df["stat_rank"] = rank_normalize(providers["statistical_outlier_score"].values)
    if "fraud_flags_count" in providers.columns:
        df["flags_rank"] = rank_normalize(providers["fraud_flags_count"].values)

    if embedding_anomaly is not None:
        df["emb_rank"] = rank_normalize(embedding_anomaly)

    # Average all rank columns
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    df["ensemble_anomaly_rank"] = df[rank_cols].mean(axis=1)

    return df


def plot_autoencoder_results(
    providers: pd.DataFrame,
    ae_scores: np.ndarray,
    history: dict,
    ensemble_df: pd.DataFrame,
    config: dict,
) -> None:
    """Generate autoencoder analysis visualizations."""
    setup_plotting(config)

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "autoencoder_training_loss", config)
    plt.close(fig)

    # 2. Reconstruction error distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(ae_scores, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Autoencoder Reconstruction Error Distribution")
    ax.set_yscale("log")
    for pct in [95, 99, 99.9]:
        val = np.percentile(ae_scores, pct)
        ax.axvline(val, color="red", linestyle="--", alpha=0.7)
        ax.text(val, ax.get_ylim()[1] * 0.5, f"p{pct}", color="red", fontsize=9, rotation=90)
    save_figure(fig, "autoencoder_reconstruction_error_dist", config)
    plt.close(fig)

    # 3. Error vs spending
    fig, ax = plt.subplots(figsize=(10, 8))
    log_spending = np.log1p(providers["total_paid"].clip(lower=0).values)
    idx = np.random.RandomState(42).choice(len(ae_scores), min(50000, len(ae_scores)), replace=False)
    ax.scatter(log_spending[idx], ae_scores[idx], s=1, alpha=0.2, rasterized=True)
    ax.set_xlabel("log(Total Spending)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Autoencoder Error vs Provider Spending")
    save_figure(fig, "autoencoder_error_vs_spending", config)
    plt.close(fig)

    # 4. Method correlation heatmap
    rank_cols = [c for c in ensemble_df.columns if c.endswith("_rank")]
    if len(rank_cols) >= 2:
        corr = ensemble_df[rank_cols].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(rank_cols)))
        ax.set_yticks(range(len(rank_cols)))
        labels = [c.replace("_rank", "").upper() for c in rank_cols]
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(len(rank_cols)):
            for j in range(len(rank_cols)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, label="Spearman Correlation")
        ax.set_title("Anomaly Method Rank Correlations")
        save_figure(fig, "anomaly_method_comparison", config)
        plt.close(fig)

    # 5. Ensemble agreement
    if "fraud_flags_count" in providers.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_1pct_ae = ae_scores >= np.percentile(ae_scores, 99)
        top_1pct_if = providers.get("isolation_forest_label", pd.Series(dtype=int)).values == -1 if "isolation_forest_label" in providers.columns else np.zeros(len(providers), dtype=bool)
        top_flags = providers["fraud_flags_count"].values >= 3

        counts = {
            "AE only": int((top_1pct_ae & ~top_1pct_if & ~top_flags).sum()),
            "IF only": int((~top_1pct_ae & top_1pct_if & ~top_flags).sum()),
            "Flags only": int((~top_1pct_ae & ~top_1pct_if & top_flags).sum()),
            "AE + IF": int((top_1pct_ae & top_1pct_if & ~top_flags).sum()),
            "AE + Flags": int((top_1pct_ae & ~top_1pct_if & top_flags).sum()),
            "IF + Flags": int((~top_1pct_ae & top_1pct_if & top_flags).sum()),
            "All three": int((top_1pct_ae & top_1pct_if & top_flags).sum()),
        }
        ax.bar(counts.keys(), counts.values(), color="steelblue", edgecolor="white")
        ax.set_ylabel("Provider Count")
        ax.set_title("Anomaly Method Agreement (top 1% AE, IF anomalies, 3+ flags)")
        plt.xticks(rotation=30, ha="right")
        for i, (k, v) in enumerate(counts.items()):
            ax.text(i, v + 10, str(v), ha="center", fontsize=9)
        save_figure(fig, "ensemble_anomaly_agreement", config)
        plt.close(fig)


def run_autoencoder_anomaly(config: Optional[dict] = None) -> pd.DataFrame:
    """Run the full autoencoder anomaly detection pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    ae_cfg = config.get("autoencoder", {})

    # Load provider features (with anomaly scores if available)
    anomaly_path = processed_dir / "provider_anomaly_scores.parquet"
    if anomaly_path.exists():
        providers = pd.read_parquet(anomaly_path)
        logger.info(f"Loaded provider anomaly scores: {providers.shape}")
    else:
        providers = pd.read_parquet(processed_dir / "provider_features.parquet")
        logger.info(f"Loaded provider features: {providers.shape}")

    # Prepare features
    X, feature_cols = prepare_features(providers)
    logger.info(f"Feature matrix: {X.shape} ({len(feature_cols)} features)")

    # Train autoencoder
    logger.info("Training autoencoder...")
    ae = ProviderAutoencoder(input_dim=X.shape[1])
    history = ae.fit(
        X,
        batch_size=ae_cfg.get("batch_size", 4096),
        epochs=ae_cfg.get("epochs", 100),
        lr=ae_cfg.get("learning_rate", 1e-3),
        weight_decay=ae_cfg.get("weight_decay", 1e-4),
        patience=ae_cfg.get("patience", 15),
    )

    # Compute reconstruction errors
    logger.info("Computing reconstruction errors for all providers...")
    ae_scores = ae.predict_reconstruction_error(X)
    logger.info(f"Reconstruction error: mean={ae_scores.mean():.6f}, "
                f"p95={np.percentile(ae_scores, 95):.6f}, "
                f"p99={np.percentile(ae_scores, 99):.6f}")

    # Extract bottleneck features
    logger.info("Extracting bottleneck features...")
    bottleneck = ae.get_bottleneck_features(X)

    # Save autoencoder scores
    ae_df = pd.DataFrame({
        "billing_npi": providers["billing_npi"].values,
        "ae_reconstruction_error": ae_scores,
    })
    ae_df.to_parquet(processed_dir / "provider_ae_anomaly.parquet", index=False)

    # Save bottleneck features
    bn_df = pd.DataFrame(bottleneck, columns=[f"ae_bottleneck_{i}" for i in range(bottleneck.shape[1])])
    bn_df.insert(0, "billing_npi", providers["billing_npi"].values)
    bn_df.to_parquet(processed_dir / "provider_ae_bottleneck.parquet", index=False)

    # Load embedding anomaly if available
    emb_anomaly = None
    emb_path = processed_dir / "provider_embedding_anomaly.parquet"
    if emb_path.exists():
        emb_df = pd.read_parquet(emb_path)
        emb_anomaly = emb_df["svd_knn_anomaly"].values

    # Build ensemble
    logger.info("Building ensemble anomaly scores...")
    ensemble_df = build_ensemble_scores(providers, ae_scores, emb_anomaly)
    ensemble_df.to_parquet(processed_dir / "provider_ensemble_anomaly.parquet", index=False)

    # Top anomalies table
    top_ae = ae_df.nlargest(100, "ae_reconstruction_error")
    save_table(top_ae, "autoencoder_top_anomalies", config)

    # Correlation table
    rank_cols = [c for c in ensemble_df.columns if c.endswith("_rank")]
    if len(rank_cols) >= 2:
        corr = ensemble_df[rank_cols].corr(method="spearman")
        save_table(corr.reset_index(), "anomaly_method_correlations", config)

    # Save model
    import torch
    model_path = root / config["paths"]["models_dir"] / "autoencoder_provider.pt"
    torch.save(ae.model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Plots
    logger.info("Generating autoencoder plots...")
    plot_autoencoder_results(providers, ae_scores, history, ensemble_df, config)

    logger.info("Autoencoder anomaly detection complete")
    return ensemble_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run autoencoder anomaly detection")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_autoencoder_anomaly(config)
