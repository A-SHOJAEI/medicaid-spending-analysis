"""Memory-efficient data loading and schema validation."""

import logging
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import load_config, get_project_root

logger = logging.getLogger("medicaid_analysis")

SCHEMA = {
    "BILLING_PROVIDER_NPI_NUM": "str",
    "SERVICING_PROVIDER_NPI_NUM": "str",
    "HCPCS_CODE": "str",
    "CLAIM_FROM_MONTH": "str",
    "TOTAL_UNIQUE_BENEFICIARIES": "Int64",
    "TOTAL_CLAIMS": "Int64",
    "TOTAL_PAID": "float64",
}


def iter_chunks(
    filepath: Optional[str] = None,
    chunksize: Optional[int] = None,
    config: Optional[dict] = None,
) -> Generator[pd.DataFrame, None, None]:
    """Yield chunks of the raw CSV file with proper dtypes.

    Args:
        filepath: Path to CSV. If None, uses config.
        chunksize: Rows per chunk. If None, uses config.
        config: Configuration dictionary.

    Yields:
        pandas DataFrame chunks.
    """
    if config is None:
        config = load_config()
    if filepath is None:
        filepath = get_project_root() / config["paths"]["raw_data"]
    if chunksize is None:
        chunksize = config["data"]["chunksize"]

    reader = pd.read_csv(
        filepath,
        chunksize=chunksize,
        dtype={
            "BILLING_PROVIDER_NPI_NUM": str,
            "SERVICING_PROVIDER_NPI_NUM": str,
            "HCPCS_CODE": str,
            "CLAIM_FROM_MONTH": str,
        },
        na_values=["", "NA", "NULL"],
    )

    for chunk in reader:
        chunk["TOTAL_UNIQUE_BENEFICIARIES"] = pd.to_numeric(
            chunk["TOTAL_UNIQUE_BENEFICIARIES"], errors="coerce"
        ).astype("Int64")
        chunk["TOTAL_CLAIMS"] = pd.to_numeric(
            chunk["TOTAL_CLAIMS"], errors="coerce"
        ).astype("Int64")
        chunk["TOTAL_PAID"] = pd.to_numeric(
            chunk["TOTAL_PAID"], errors="coerce"
        )
        yield chunk


def load_parquet(
    name: str, config: Optional[dict] = None
) -> pd.DataFrame:
    """Load a processed parquet file.

    Args:
        name: Name of the parquet file (without extension).
        config: Configuration dictionary.

    Returns:
        pandas DataFrame.
    """
    if config is None:
        config = load_config()
    path = get_project_root() / config["paths"]["processed_dir"] / f"{name}.parquet"
    return pd.read_parquet(path)


def load_parquet_lazy(
    name: str, config: Optional[dict] = None
):
    """Load a processed parquet file using polars lazy frame.

    Args:
        name: Name of the parquet file (without extension).
        config: Configuration dictionary.

    Returns:
        polars LazyFrame.
    """
    import polars as pl
    if config is None:
        config = load_config()
    path = get_project_root() / config["paths"]["processed_dir"] / f"{name}.parquet"
    return pl.scan_parquet(path)


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate DataFrame columns match expected schema.

    Args:
        df: DataFrame to validate.

    Returns:
        True if schema matches.

    Raises:
        ValueError: If required columns are missing.
    """
    expected = set(SCHEMA.keys())
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = actual - expected
    if extra:
        logger.warning(f"Extra columns found: {extra}")
    return True


def get_sample(
    n: int = 10000,
    filepath: Optional[str] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Read the first n rows for quick exploration.

    Args:
        n: Number of rows to read.
        filepath: Path to CSV.
        config: Configuration dictionary.

    Returns:
        pandas DataFrame with n rows.
    """
    if config is None:
        config = load_config()
    if filepath is None:
        filepath = get_project_root() / config["paths"]["raw_data"]

    return pd.read_csv(
        filepath,
        nrows=n,
        dtype={
            "BILLING_PROVIDER_NPI_NUM": str,
            "SERVICING_PROVIDER_NPI_NUM": str,
            "HCPCS_CODE": str,
            "CLAIM_FROM_MONTH": str,
        },
    )
