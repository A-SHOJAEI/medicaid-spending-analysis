"""Phase 7B: Variational Autoencoder (VAE) for provider anomaly detection.

Trains a β-VAE with cyclical KL annealing to learn disentangled latent
representations of provider billing patterns.  Anomalies are detected via
a combined score of reconstruction loss and KL divergence from the prior.

Key innovations:
    - β-VAE with cyclical annealing for disentangled features
    - Combined ELBO-based anomaly scoring
    - Latent space traversals for interpretability
    - Comparison with deterministic autoencoder
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table,
)

logger = setup_logging()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────


class BetaVAE(nn.Module):
    """β-VAE with configurable encoder/decoder architecture."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """ELBO loss: reconstruction + β * KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="none").sum(dim=1)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return recon_loss, kl_loss, (recon_loss + beta * kl_loss).mean()


# ── Training ──────────────────────────────────────────────────────────


def cyclical_beta_schedule(epoch: int, n_epochs: int, n_cycles: int = 4,
                            ratio: float = 0.5, beta_max: float = 4.0) -> float:
    """Cyclical annealing schedule for β (Fu et al., 2019)."""
    cycle_len = n_epochs / n_cycles
    pos_in_cycle = (epoch % cycle_len) / cycle_len
    if pos_in_cycle < ratio:
        return beta_max * pos_in_cycle / ratio
    return beta_max


def train_vae(model: BetaVAE, train_loader: DataLoader,
              val_loader: DataLoader, config: dict,
              device: torch.device) -> dict:
    """Train the VAE with cyclical β-annealing and early stopping."""
    vae_cfg = config.get("vae", {})
    n_epochs = vae_cfg.get("epochs", 100)
    lr = vae_cfg.get("learning_rate", 1e-3)
    patience = vae_cfg.get("patience", 15)
    beta_max = vae_cfg.get("beta_max", 4.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "train_recon": [],
               "train_kl": [], "beta": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        beta = cyclical_beta_schedule(epoch, n_epochs, beta_max=beta_max)
        model.train()
        epoch_loss, epoch_recon, epoch_kl, n_batches = 0, 0, 0, 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            recon, mu, logvar, _ = model(batch_x)
            recon_l, kl_l, loss = vae_loss(recon, batch_x, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon += recon_l.mean().item()
            epoch_kl += kl_l.mean().item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss_total = 0
        val_batches = 0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon, mu, logvar, _ = model(batch_x)
                _, _, vloss = vae_loss(recon, batch_x, mu, logvar, beta)
                val_loss_total += vloss.item()
                val_batches += 1

        train_loss = epoch_loss / n_batches
        val_loss = val_loss_total / val_batches
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon"].append(epoch_recon / n_batches)
        history["train_kl"].append(epoch_kl / n_batches)
        history["beta"].append(beta)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(f"  Epoch {epoch+1}/{n_epochs} — train={train_loss:.4f} "
                        f"val={val_loss:.4f} β={beta:.2f} "
                        f"recon={epoch_recon/n_batches:.4f} kl={epoch_kl/n_batches:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ── Anomaly Scoring ───────────────────────────────────────────────────


@torch.no_grad()
def compute_anomaly_scores(model: BetaVAE, data_loader: DataLoader,
                            device: torch.device) -> dict:
    """Compute per-sample reconstruction error, KL divergence, and combined score."""
    model.eval()
    all_recon, all_kl, all_mu, all_z = [], [], [], []

    for (batch_x,) in data_loader:
        batch_x = batch_x.to(device)
        recon, mu, logvar, z = model(batch_x)
        recon_err = nn.functional.mse_loss(recon, batch_x, reduction="none").sum(dim=1)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        all_recon.append(recon_err.cpu().numpy())
        all_kl.append(kl_div.cpu().numpy())
        all_mu.append(mu.cpu().numpy())
        all_z.append(z.cpu().numpy())

    return {
        "recon_error": np.concatenate(all_recon),
        "kl_divergence": np.concatenate(all_kl),
        "mu": np.concatenate(all_mu),
        "z": np.concatenate(all_z),
    }


# ── Visualization ─────────────────────────────────────────────────────


def plot_training_history(history: dict, config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train ELBO", color="#2196F3")
    axes[0].plot(history["val_loss"], label="Val ELBO", color="#FF5722")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ELBO Loss")
    axes[0].set_title("Training Convergence")
    axes[0].legend()

    axes[1].plot(history["train_recon"], label="Recon Loss", color="#4CAF50")
    axes[1].plot(history["train_kl"], label="KL Divergence", color="#9C27B0")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].set_title("Reconstruction vs KL Divergence")
    axes[1].legend()

    axes[2].plot(history["beta"], color="#FF9800", lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("β")
    axes[2].set_title("Cyclical β-Annealing Schedule")

    fig.suptitle("β-VAE Training — Cyclical Annealing", fontsize=14, fontweight="bold")
    save_figure(fig, "vae_training_history", config)


def plot_latent_space(mu: np.ndarray, scores: np.ndarray, config: dict) -> None:
    """2D projection of latent space colored by anomaly score."""
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Use first 2 latent dims
    if mu.shape[1] >= 2:
        x, y = mu[:, 0], mu[:, 1]
    else:
        x, y = mu[:, 0], np.zeros(len(mu))

    sc = axes[0].scatter(x, y, c=scores, cmap="hot", alpha=0.3, s=3, edgecolors="none")
    plt.colorbar(sc, ax=axes[0], label="ELBO Anomaly Score")
    axes[0].set_xlabel("z₁ (μ)")
    axes[0].set_ylabel("z₂ (μ)")
    axes[0].set_title("Latent Space — Colored by Anomaly Score")

    # KDE of latent dims
    for i in range(min(8, mu.shape[1])):
        axes[1].hist(mu[:, i], bins=100, alpha=0.5, density=True,
                     label=f"z{i+1}")
    axes[1].set_xlabel("Latent Value")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Latent Dimension Distributions")
    axes[1].legend(fontsize=8, ncol=2)

    fig.suptitle("β-VAE Latent Space Analysis", fontsize=14, fontweight="bold")
    save_figure(fig, "vae_latent_space", config)


def plot_anomaly_distribution(recon_err: np.ndarray, kl_div: np.ndarray,
                               config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(recon_err, bins=200, color="#2196F3", alpha=0.7, density=True)
    p99 = np.percentile(recon_err, 99)
    axes[0].axvline(p99, color="red", ls="--", label=f"99th pctile = {p99:.2f}")
    axes[0].set_xlabel("Reconstruction Error")
    axes[0].set_title("Reconstruction Error Distribution")
    axes[0].legend()
    axes[0].set_xlim(0, np.percentile(recon_err, 99.5))

    axes[1].hist(kl_div, bins=200, color="#4CAF50", alpha=0.7, density=True)
    p99_kl = np.percentile(kl_div, 99)
    axes[1].axvline(p99_kl, color="red", ls="--", label=f"99th pctile = {p99_kl:.2f}")
    axes[1].set_xlabel("KL Divergence")
    axes[1].set_title("KL Divergence Distribution")
    axes[1].legend()
    axes[1].set_xlim(0, np.percentile(kl_div, 99.5))

    combined = recon_err / recon_err.std() + kl_div / kl_div.std()
    axes[2].hist(combined, bins=200, color="#FF9800", alpha=0.7, density=True)
    p99_c = np.percentile(combined, 99)
    axes[2].axvline(p99_c, color="red", ls="--", label=f"99th pctile = {p99_c:.2f}")
    axes[2].set_xlabel("Combined ELBO Score")
    axes[2].set_title("Combined Anomaly Score")
    axes[2].legend()
    axes[2].set_xlim(0, np.percentile(combined, 99.5))

    fig.suptitle("β-VAE Anomaly Score Distributions", fontsize=14, fontweight="bold")
    save_figure(fig, "vae_anomaly_distributions", config)


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_vae_analysis(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load provider features
    df = pd.read_parquet(processed_dir / "provider_features.parquet")
    logger.info(f"Loaded {len(df)} providers")

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != "billing_npi"]
    X_raw = df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)
    logger.info(f"Feature matrix: {X.shape}")

    # Train/val split
    rng = np.random.RandomState(config["analysis"]["random_seed"])
    perm = rng.permutation(len(X))
    n_val = int(0.15 * len(X))
    X_train = X[perm[n_val:]]
    X_val = X[perm[:n_val]]

    vae_cfg = config.get("vae", {})
    batch_size = vae_cfg.get("batch_size", 4096)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Build model
    input_dim = X.shape[1]
    hidden_dims = vae_cfg.get("architecture", [128, 64])
    latent_dim = vae_cfg.get("latent_dim", 16)
    dropout = vae_cfg.get("dropout", 0.1)

    model = BetaVAE(input_dim, hidden_dims, latent_dim, dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"β-VAE: {n_params:,} parameters, latent_dim={latent_dim}")

    # Train
    logger.info("Training β-VAE with cyclical annealing...")
    history = train_vae(model, train_loader, val_loader, config, device)

    # Full dataset scoring
    full_loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                             batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    scores = compute_anomaly_scores(model, full_loader, device)

    recon_err = scores["recon_error"]
    kl_div = scores["kl_divergence"]
    combined = recon_err / (recon_err.std() + 1e-10) + kl_div / (kl_div.std() + 1e-10)

    # Save results
    result_df = pd.DataFrame({
        "billing_npi": df["billing_npi"].values,
        "vae_recon_error": recon_err,
        "vae_kl_divergence": kl_div,
        "vae_anomaly_score": combined,
        "vae_anomaly_rank": pd.Series(combined).rank(ascending=False, method="min").values,
    })
    result_df.to_parquet(processed_dir / "provider_vae_scores.parquet", index=False)
    logger.info(f"Saved VAE scores for {len(result_df)} providers")

    # Save latent representations
    latent_df = pd.DataFrame(
        scores["mu"],
        columns=[f"vae_z{i}" for i in range(scores["mu"].shape[1])]
    )
    latent_df.insert(0, "billing_npi", df["billing_npi"].values)
    latent_df.to_parquet(processed_dir / "provider_vae_latent.parquet", index=False)

    # Top anomalies table
    top_idx = np.argsort(combined)[::-1][:50]
    top_df = df.iloc[top_idx][["billing_npi"] + feature_cols[:6]].copy()
    top_df["vae_anomaly_score"] = combined[top_idx]
    top_df["vae_recon_error"] = recon_err[top_idx]
    top_df["vae_kl_divergence"] = kl_div[top_idx]
    save_table(top_df, "vae_top_anomalies", config)

    # Visualizations
    plot_training_history(history, config)
    plot_latent_space(scores["mu"], combined, config)
    plot_anomaly_distribution(recon_err, kl_div, config)

    # Summary
    summary = {
        "n_providers": len(df),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "n_parameters": n_params,
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "mean_recon_error": float(recon_err.mean()),
        "mean_kl_divergence": float(kl_div.mean()),
        "anomaly_99th_percentile": float(np.percentile(combined, 99)),
        "anomaly_95th_percentile": float(np.percentile(combined, 95)),
    }
    tables_dir = root / config["paths"]["tables_dir"]
    with open(tables_dir / "vae_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"VAE summary: {summary}")

    # Save model
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / "beta_vae.pt")

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("Phase 7B: β-VAE Anomaly Detection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7B: VAE Anomaly Detection")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_vae_analysis(config)
