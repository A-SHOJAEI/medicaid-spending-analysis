"""Shared helpers, logging, and plotting defaults."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml.

    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger.

    Args:
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("medicaid_analysis")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_plotting(config: Optional[dict] = None) -> None:
    """Configure matplotlib/seaborn defaults for publication-quality plots.

    Args:
        config: Configuration dictionary with plotting parameters.
    """
    if config is None:
        config = load_config()
    plot_cfg = config.get("plotting", {})

    plt.style.use(plot_cfg.get("style", "seaborn-v0_8-whitegrid"))
    sns.set_context("paper", font_scale=plot_cfg.get("font_scale", 1.2))
    plt.rcParams.update({
        "figure.dpi": plot_cfg.get("dpi", 300),
        "savefig.dpi": plot_cfg.get("dpi", 300),
        "figure.figsize": plot_cfg.get("figsize_default", [12, 8]),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig: plt.Figure, name: str, config: Optional[dict] = None) -> None:
    """Save figure in both PNG and PDF formats.

    Args:
        fig: Matplotlib figure object.
        name: Base filename (without extension).
        config: Configuration dictionary.
    """
    if config is None:
        config = load_config()
    fig_dir = get_project_root() / config["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        filepath = fig_dir / f"{name}.{ext}"
        fig.savefig(filepath, bbox_inches="tight", dpi=config["plotting"]["dpi"])
    plt.close(fig)


def save_table(df, name: str, config: Optional[dict] = None) -> None:
    """Save DataFrame as CSV to the tables output directory.

    Args:
        df: pandas DataFrame.
        name: Filename (without extension).
        config: Configuration dictionary.
    """
    if config is None:
        config = load_config()
    table_dir = get_project_root() / config["paths"]["tables_dir"]
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / f"{name}.csv", index=True)


def format_currency(value: float) -> str:
    """Format a number as US currency string.

    Args:
        value: Numeric value.

    Returns:
        Formatted string like '$1.23B' or '$456.7M'.
    """
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val/1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val/1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}${abs_val/1e3:.1f}K"
    else:
        return f"{sign}${abs_val:.2f}"


def format_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        value: Numeric value.

    Returns:
        Formatted string.
    """
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= 1e9:
        return f"{sign}{abs_val/1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}{abs_val/1e6:.1f}M"
    elif abs_val >= 1e3:
        return f"{sign}{abs_val/1e3:.1f}K"
    else:
        return f"{sign}{abs_val:.0f}"
