"""Data cleaning: missing values, type casting, deduplication, consistency checks."""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root

logger = logging.getLogger("medicaid_analysis")


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning operations to a single chunk.

    Args:
        df: Raw DataFrame chunk.

    Returns:
        Cleaned DataFrame.
    """
    # Remove fully duplicate rows
    df = df.drop_duplicates()

    # Ensure numeric columns are properly typed
    df["TOTAL_PAID"] = pd.to_numeric(df["TOTAL_PAID"], errors="coerce")
    df["TOTAL_CLAIMS"] = pd.to_numeric(df["TOTAL_CLAIMS"], errors="coerce").astype("Int64")
    df["TOTAL_UNIQUE_BENEFICIARIES"] = pd.to_numeric(
        df["TOTAL_UNIQUE_BENEFICIARIES"], errors="coerce"
    ).astype("Int64")

    # Strip whitespace from string columns
    for col in ["BILLING_PROVIDER_NPI_NUM", "SERVICING_PROVIDER_NPI_NUM", "HCPCS_CODE"]:
        df[col] = df[col].str.strip()

    # Validate NPI format (10 digits)
    df["VALID_BILLING_NPI"] = df["BILLING_PROVIDER_NPI_NUM"].str.match(r"^\d{10}$", na=False)
    df["VALID_SERVICING_NPI"] = df["SERVICING_PROVIDER_NPI_NUM"].str.match(r"^\d{10}$", na=True)

    # Flag suspicious records
    df["FLAG_NEGATIVE_PAID"] = df["TOTAL_PAID"] < 0
    df["FLAG_ZERO_CLAIMS"] = df["TOTAL_CLAIMS"] == 0
    df["FLAG_HIGH_PAID_PER_CLAIM"] = False

    mask = df["TOTAL_CLAIMS"] > 0
    paid_per_claim = df.loc[mask, "TOTAL_PAID"] / df.loc[mask, "TOTAL_CLAIMS"]
    df.loc[mask, "FLAG_HIGH_PAID_PER_CLAIM"] = paid_per_claim > 100000

    return df


def assess_data_quality(config: Optional[dict] = None) -> pd.DataFrame:
    """Assess data quality across the full dataset via chunked processing.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with quality metrics.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    quality_metrics = []
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf)
        year = pf.stem.split("_")[1]

        metrics = {
            "year": year,
            "total_rows": len(df),
            "null_billing_npi": df["BILLING_PROVIDER_NPI_NUM"].isna().sum(),
            "null_servicing_npi": (df["SERVICING_PROVIDER_NPI_NUM"] == "").sum(),
            "null_hcpcs": df["HCPCS_CODE"].isna().sum(),
            "negative_paid_count": (df["TOTAL_PAID"] < 0).sum(),
            "negative_paid_total": df.loc[df["TOTAL_PAID"] < 0, "TOTAL_PAID"].sum(),
            "zero_paid_count": (df["TOTAL_PAID"] == 0).sum(),
            "total_paid": df["TOTAL_PAID"].sum(),
            "total_claims": df["TOTAL_CLAIMS"].sum(),
            "unique_billing_npi": df["BILLING_PROVIDER_NPI_NUM"].nunique(),
            "unique_hcpcs": df["HCPCS_CODE"].nunique(),
            "unique_months": df["CLAIM_FROM_MONTH"].nunique(),
        }
        quality_metrics.append(metrics)
        del df

    return pd.DataFrame(quality_metrics)
