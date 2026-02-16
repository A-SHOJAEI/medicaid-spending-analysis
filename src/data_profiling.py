"""Phase 1: Data profiling and quality assessment via chunked processing."""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root, setup_logging, format_currency, format_number

logger = setup_logging()


def profile_dataset(config: dict) -> dict:
    """Run full chunked profiling of the raw CSV.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary of profiling results.
    """
    filepath = get_project_root() / config["paths"]["raw_data"]
    chunksize = config["data"]["chunksize"]
    file_size_gb = os.path.getsize(filepath) / (1024 ** 3)

    logger.info(f"Profiling {filepath} ({file_size_gb:.2f} GB)")

    # Accumulators
    total_rows = 0
    null_counts = Counter()
    paid_stats = {"sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf"), "neg_count": 0, "zero_count": 0}
    claims_stats = {"sum": 0, "min": float("inf"), "max": float("-inf")}
    benef_stats = {"sum": 0, "min": float("inf"), "max": float("-inf")}

    billing_npi_set = set()
    servicing_npi_set = set()
    hcpcs_counter = Counter()
    month_counter = Counter()
    billing_npi_counter = Counter()

    # Per-month spending
    monthly_spending = Counter()
    monthly_claims = Counter()
    monthly_beneficiaries = Counter()

    # Per-HCPCS spending
    hcpcs_spending = Counter()
    hcpcs_claims = Counter()

    # Paid per claim distribution bins
    paid_per_claim_hist = Counter()

    # Track billing=servicing match
    npi_match_count = 0
    servicing_null_count = 0

    chunk_num = 0
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
        chunk_num += 1
        n = len(chunk)
        total_rows += n

        if chunk_num % 10 == 0:
            logger.info(f"  Chunk {chunk_num}: {total_rows:,} rows processed")

        # Null counts
        for col in chunk.columns:
            null_counts[col] += int(chunk[col].isna().sum())

        # Numeric stats
        paid = pd.to_numeric(chunk["TOTAL_PAID"], errors="coerce")
        claims = pd.to_numeric(chunk["TOTAL_CLAIMS"], errors="coerce")
        benef = pd.to_numeric(chunk["TOTAL_UNIQUE_BENEFICIARIES"], errors="coerce")

        paid_valid = paid.dropna()
        paid_stats["sum"] += float(paid_valid.sum())
        paid_stats["sum_sq"] += float((paid_valid ** 2).sum())
        paid_stats["min"] = min(paid_stats["min"], float(paid_valid.min()) if len(paid_valid) > 0 else paid_stats["min"])
        paid_stats["max"] = max(paid_stats["max"], float(paid_valid.max()) if len(paid_valid) > 0 else paid_stats["max"])
        paid_stats["neg_count"] += int((paid_valid < 0).sum())
        paid_stats["zero_count"] += int((paid_valid == 0).sum())

        claims_valid = claims.dropna()
        claims_stats["sum"] += int(claims_valid.sum())
        claims_stats["min"] = min(claims_stats["min"], int(claims_valid.min()) if len(claims_valid) > 0 else claims_stats["min"])
        claims_stats["max"] = max(claims_stats["max"], int(claims_valid.max()) if len(claims_valid) > 0 else claims_stats["max"])

        benef_valid = benef.dropna()
        benef_stats["sum"] += int(benef_valid.sum())
        benef_stats["min"] = min(benef_stats["min"], int(benef_valid.min()) if len(benef_valid) > 0 else benef_stats["min"])
        benef_stats["max"] = max(benef_stats["max"], int(benef_valid.max()) if len(benef_valid) > 0 else benef_stats["max"])

        # Categorical counters
        hcpcs_counter.update(chunk["HCPCS_CODE"].dropna().values)
        month_counter.update(chunk["CLAIM_FROM_MONTH"].dropna().values)
        billing_npi_set.update(chunk["BILLING_PROVIDER_NPI_NUM"].dropna().values)
        servicing_npi_set.update(chunk["SERVICING_PROVIDER_NPI_NUM"].dropna().values)

        # Top billing NPIs (by row count)
        billing_npi_counter.update(chunk["BILLING_PROVIDER_NPI_NUM"].dropna().values)

        # Monthly aggregations
        for month, group in chunk.groupby("CLAIM_FROM_MONTH"):
            monthly_spending[month] += float(pd.to_numeric(group["TOTAL_PAID"], errors="coerce").sum())
            monthly_claims[month] += int(pd.to_numeric(group["TOTAL_CLAIMS"], errors="coerce").sum())
            monthly_beneficiaries[month] += int(pd.to_numeric(group["TOTAL_UNIQUE_BENEFICIARIES"], errors="coerce").sum())

        # Per-HCPCS aggregations
        for code, group in chunk.groupby("HCPCS_CODE"):
            hcpcs_spending[code] += float(pd.to_numeric(group["TOTAL_PAID"], errors="coerce").sum())
            hcpcs_claims[code] += int(pd.to_numeric(group["TOTAL_CLAIMS"], errors="coerce").sum())

        # Billing = Servicing match
        both_present = chunk.dropna(subset=["BILLING_PROVIDER_NPI_NUM", "SERVICING_PROVIDER_NPI_NUM"])
        npi_match_count += int((both_present["BILLING_PROVIDER_NPI_NUM"] == both_present["SERVICING_PROVIDER_NPI_NUM"]).sum())
        servicing_null_count += int(chunk["SERVICING_PROVIDER_NPI_NUM"].isna().sum())

    # Compute derived stats
    paid_mean = paid_stats["sum"] / total_rows
    paid_var = (paid_stats["sum_sq"] / total_rows) - (paid_mean ** 2)
    paid_std = np.sqrt(max(0, paid_var))

    results = {
        "file_size_gb": round(file_size_gb, 2),
        "total_rows": total_rows,
        "total_chunks": chunk_num,
        "columns": list(pd.read_csv(filepath, nrows=0).columns),
        "null_counts": dict(null_counts),
        "null_rates": {k: round(v / total_rows * 100, 4) for k, v in null_counts.items()},
        "paid_stats": {
            "total": paid_stats["sum"],
            "mean": paid_mean,
            "std": paid_std,
            "min": paid_stats["min"],
            "max": paid_stats["max"],
            "negative_rows": paid_stats["neg_count"],
            "zero_rows": paid_stats["zero_count"],
        },
        "claims_stats": {
            "total": claims_stats["sum"],
            "min": claims_stats["min"],
            "max": claims_stats["max"],
        },
        "beneficiary_stats": {
            "total": benef_stats["sum"],
            "min": benef_stats["min"],
            "max": benef_stats["max"],
        },
        "cardinality": {
            "billing_npi": len(billing_npi_set),
            "servicing_npi": len(servicing_npi_set),
            "hcpcs_codes": len(hcpcs_counter),
            "months": len(month_counter),
        },
        "date_range": {
            "min": min(month_counter.keys()),
            "max": max(month_counter.keys()),
            "months_covered": sorted(month_counter.keys()),
        },
        "npi_relationship": {
            "billing_equals_servicing": npi_match_count,
            "servicing_null": servicing_null_count,
            "total_rows": total_rows,
        },
        "top_20_hcpcs_by_spending": dict(
            sorted(hcpcs_spending.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
        "top_20_hcpcs_by_frequency": dict(hcpcs_counter.most_common(20)),
        "top_20_billing_npi_by_rows": dict(billing_npi_counter.most_common(20)),
        "monthly_spending": dict(sorted(monthly_spending.items())),
        "monthly_claims": dict(sorted(monthly_claims.items())),
    }

    return results


def save_profiling_results(results: dict, config: dict) -> None:
    """Save profiling results as JSON and summary tables.

    Args:
        results: Profiling results dictionary.
        config: Configuration dictionary.
    """
    root = get_project_root()
    tables_dir = root / config["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON (convert numpy types)
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(tables_dir / "profiling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

    # Monthly spending table
    monthly_df = pd.DataFrame({
        "month": list(results["monthly_spending"].keys()),
        "total_paid": list(results["monthly_spending"].values()),
        "total_claims": [results["monthly_claims"].get(m, 0) for m in results["monthly_spending"].keys()],
    })
    monthly_df.to_csv(tables_dir / "monthly_spending.csv", index=False)

    # Top HCPCS
    hcpcs_df = pd.DataFrame([
        {"hcpcs_code": k, "total_spending": v}
        for k, v in results["top_20_hcpcs_by_spending"].items()
    ])
    hcpcs_df.to_csv(tables_dir / "top_hcpcs_by_spending.csv", index=False)

    logger.info(f"Profiling results saved to {tables_dir}")


def convert_to_parquet(config: dict) -> None:
    """Convert raw CSV to partitioned Parquet files by year.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    filepath = root / config["paths"]["raw_data"]
    output_dir = root / config["paths"]["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    chunksize = config["data"]["chunksize"]
    logger.info("Converting CSV to Parquet (partitioned by year)...")

    # Collect by year
    writers = {}
    schema = pa.schema([
        ("BILLING_PROVIDER_NPI_NUM", pa.string()),
        ("SERVICING_PROVIDER_NPI_NUM", pa.string()),
        ("HCPCS_CODE", pa.string()),
        ("CLAIM_FROM_MONTH", pa.string()),
        ("YEAR", pa.int32()),
        ("MONTH_NUM", pa.int32()),
        ("TOTAL_UNIQUE_BENEFICIARIES", pa.int64()),
        ("TOTAL_CLAIMS", pa.int64()),
        ("TOTAL_PAID", pa.float64()),
        ("PAID_PER_CLAIM", pa.float64()),
        ("PAID_PER_BENEFICIARY", pa.float64()),
    ])

    chunk_num = 0
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
        chunk_num += 1
        if chunk_num % 10 == 0:
            logger.info(f"  Parquet conversion chunk {chunk_num}")

        # Type conversions
        chunk["TOTAL_UNIQUE_BENEFICIARIES"] = pd.to_numeric(
            chunk["TOTAL_UNIQUE_BENEFICIARIES"], errors="coerce"
        ).fillna(0).astype(np.int64)
        chunk["TOTAL_CLAIMS"] = pd.to_numeric(
            chunk["TOTAL_CLAIMS"], errors="coerce"
        ).fillna(0).astype(np.int64)
        chunk["TOTAL_PAID"] = pd.to_numeric(
            chunk["TOTAL_PAID"], errors="coerce"
        ).fillna(0.0)

        # Derived columns
        chunk["YEAR"] = chunk["CLAIM_FROM_MONTH"].str[:4].astype(int)
        chunk["MONTH_NUM"] = chunk["CLAIM_FROM_MONTH"].str[5:7].astype(int)
        chunk["PAID_PER_CLAIM"] = np.where(
            chunk["TOTAL_CLAIMS"] > 0,
            chunk["TOTAL_PAID"] / chunk["TOTAL_CLAIMS"],
            0.0,
        )
        chunk["PAID_PER_BENEFICIARY"] = np.where(
            chunk["TOTAL_UNIQUE_BENEFICIARIES"] > 0,
            chunk["TOTAL_PAID"] / chunk["TOTAL_UNIQUE_BENEFICIARIES"],
            0.0,
        )

        # Fill NaN servicing NPI
        chunk["SERVICING_PROVIDER_NPI_NUM"] = chunk["SERVICING_PROVIDER_NPI_NUM"].fillna("")

        # Write by year
        for year, group in chunk.groupby("YEAR"):
            year_file = output_dir / f"medicaid_{year}.parquet"
            table = pa.Table.from_pandas(group, schema=schema, preserve_index=False)
            if year not in writers:
                writers[year] = pq.ParquetWriter(str(year_file), schema, compression="snappy")
            writers[year].write_table(table)

    for w in writers.values():
        w.close()

    # Verify
    for year_file in sorted(output_dir.glob("medicaid_*.parquet")):
        pf = pq.read_metadata(year_file)
        logger.info(f"  {year_file.name}: {pf.num_rows:,} rows")

    logger.info("Parquet conversion complete.")


def main():
    """Run the full profiling pipeline."""
    parser = argparse.ArgumentParser(description="Profile the Medicaid provider spending dataset")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profiling, only convert to parquet")
    parser.add_argument("--skip-parquet", action="store_true", help="Skip parquet conversion")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if not args.skip_profile:
        results = profile_dataset(config)
        save_profiling_results(results, config)

        # Print summary
        logger.info("=" * 60)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"File size: {results['file_size_gb']} GB")
        logger.info(f"Total rows: {results['total_rows']:,}")
        logger.info(f"Columns: {results['columns']}")
        logger.info(f"Date range: {results['date_range']['min']} to {results['date_range']['max']}")
        logger.info(f"Unique billing NPIs: {results['cardinality']['billing_npi']:,}")
        logger.info(f"Unique servicing NPIs: {results['cardinality']['servicing_npi']:,}")
        logger.info(f"Unique HCPCS codes: {results['cardinality']['hcpcs_codes']:,}")
        logger.info(f"Total spending: {format_currency(results['paid_stats']['total'])}")
        logger.info(f"Total claims: {format_number(results['claims_stats']['total'])}")
        logger.info(f"Negative payment rows: {results['paid_stats']['negative_rows']:,}")
        logger.info(f"Zero payment rows: {results['paid_stats']['zero_rows']:,}")
        for col, rate in results["null_rates"].items():
            logger.info(f"  {col}: {rate}% null")

    if not args.skip_parquet:
        convert_to_parquet(config)


if __name__ == "__main__":
    main()
