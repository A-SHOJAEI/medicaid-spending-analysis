"""Derived features, ratios, rolling aggregates, and flags."""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root

logger = logging.getLogger("medicaid_analysis")


def build_provider_features(config: Optional[dict] = None) -> pd.DataFrame:
    """Build provider-level feature vectors from parquet files.

    Aggregates across all years to create one row per billing NPI with
    features suitable for clustering and anomaly detection.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with one row per billing NPI.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Accumulate provider-level stats across all parquet files
    provider_agg = {}
    provider_years = {}
    provider_months = {}
    provider_hcpcs = {}
    provider_servicing = {}

    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        logger.info(f"Processing {pf.name} for provider features...")
        df = pd.read_parquet(pf, columns=[
            "BILLING_PROVIDER_NPI_NUM", "SERVICING_PROVIDER_NPI_NUM",
            "HCPCS_CODE", "YEAR", "MONTH_NUM",
            "TOTAL_CLAIMS", "TOTAL_PAID", "TOTAL_UNIQUE_BENEFICIARIES",
        ])

        for npi, group in df.groupby("BILLING_PROVIDER_NPI_NUM"):
            if npi not in provider_agg:
                provider_agg[npi] = {
                    "total_paid": 0.0, "total_claims": 0, "total_beneficiaries": 0,
                    "row_count": 0, "neg_paid_count": 0,
                }
                provider_years[npi] = set()
                provider_months[npi] = set()
                provider_hcpcs[npi] = set()
                provider_servicing[npi] = set()

            provider_agg[npi]["total_paid"] += group["TOTAL_PAID"].sum()
            provider_agg[npi]["total_claims"] += group["TOTAL_CLAIMS"].sum()
            provider_agg[npi]["total_beneficiaries"] += group["TOTAL_UNIQUE_BENEFICIARIES"].sum()
            provider_agg[npi]["row_count"] += len(group)
            provider_agg[npi]["neg_paid_count"] += (group["TOTAL_PAID"] < 0).sum()

            provider_years[npi].update(group["YEAR"].unique())
            provider_months[npi].update(
                group.apply(lambda r: f"{r['YEAR']}-{r['MONTH_NUM']:02d}", axis=1).unique()
            )
            provider_hcpcs[npi].update(group["HCPCS_CODE"].dropna().unique())
            svc = group["SERVICING_PROVIDER_NPI_NUM"]
            svc_valid = svc[svc != ""]
            provider_servicing[npi].update(svc_valid.unique())

        del df

    # Build feature DataFrame
    records = []
    for npi, agg in provider_agg.items():
        paid = agg["total_paid"]
        claims = agg["total_claims"]
        bene = agg["total_beneficiaries"]

        records.append({
            "billing_npi": npi,
            "total_paid": paid,
            "total_claims": claims,
            "total_beneficiaries": bene,
            "row_count": agg["row_count"],
            "paid_per_claim": paid / claims if claims > 0 else 0,
            "paid_per_beneficiary": paid / bene if bene > 0 else 0,
            "claims_per_beneficiary": claims / bene if bene > 0 else 0,
            "n_years_active": len(provider_years[npi]),
            "n_months_active": len(provider_months[npi]),
            "n_unique_hcpcs": len(provider_hcpcs[npi]),
            "n_servicing_npis": len(provider_servicing[npi]),
            "has_negative_payments": agg["neg_paid_count"] > 0,
            "neg_paid_ratio": agg["neg_paid_count"] / agg["row_count"],
            "self_servicing": npi in provider_servicing[npi],
        })

    features_df = pd.DataFrame(records)

    # Save
    output_path = root / config["paths"]["processed_dir"] / "provider_features.parquet"
    features_df.to_parquet(output_path, index=False)
    logger.info(f"Provider features saved: {len(features_df):,} providers, {output_path}")

    return features_df


def build_monthly_time_series(config: Optional[dict] = None) -> pd.DataFrame:
    """Build monthly national time series from parquet files.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with monthly aggregates.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    monthly_data = []
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=[
            "CLAIM_FROM_MONTH", "TOTAL_CLAIMS", "TOTAL_PAID",
            "TOTAL_UNIQUE_BENEFICIARIES", "BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE",
        ])
        monthly = df.groupby("CLAIM_FROM_MONTH").agg(
            total_paid=("TOTAL_PAID", "sum"),
            total_claims=("TOTAL_CLAIMS", "sum"),
            total_beneficiaries=("TOTAL_UNIQUE_BENEFICIARIES", "sum"),
            unique_providers=("BILLING_PROVIDER_NPI_NUM", "nunique"),
            unique_hcpcs=("HCPCS_CODE", "nunique"),
            record_count=("TOTAL_PAID", "count"),
        ).reset_index()
        monthly_data.append(monthly)
        del df

    ts = pd.concat(monthly_data, ignore_index=True)
    ts = ts.groupby("CLAIM_FROM_MONTH").sum(numeric_only=True).reset_index()
    ts["date"] = pd.to_datetime(ts["CLAIM_FROM_MONTH"] + "-01")
    ts = ts.sort_values("date").reset_index(drop=True)
    ts["paid_per_claim"] = ts["total_paid"] / ts["total_claims"]

    output_path = root / config["paths"]["processed_dir"] / "monthly_time_series.parquet"
    ts.to_parquet(output_path, index=False)
    logger.info(f"Monthly time series saved: {len(ts)} months")

    return ts


def build_hcpcs_features(config: Optional[dict] = None) -> pd.DataFrame:
    """Build HCPCS-level aggregate features.

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with one row per HCPCS code.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    hcpcs_agg = {}

    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=[
            "HCPCS_CODE", "YEAR", "TOTAL_CLAIMS", "TOTAL_PAID",
            "TOTAL_UNIQUE_BENEFICIARIES", "BILLING_PROVIDER_NPI_NUM",
        ])
        for code, group in df.groupby("HCPCS_CODE"):
            if code not in hcpcs_agg:
                hcpcs_agg[code] = {
                    "total_paid": 0.0, "total_claims": 0, "total_beneficiaries": 0,
                    "provider_set": set(), "year_set": set(), "row_count": 0,
                }
            hcpcs_agg[code]["total_paid"] += group["TOTAL_PAID"].sum()
            hcpcs_agg[code]["total_claims"] += group["TOTAL_CLAIMS"].sum()
            hcpcs_agg[code]["total_beneficiaries"] += group["TOTAL_UNIQUE_BENEFICIARIES"].sum()
            hcpcs_agg[code]["provider_set"].update(group["BILLING_PROVIDER_NPI_NUM"].unique())
            hcpcs_agg[code]["year_set"].update(group["YEAR"].unique())
            hcpcs_agg[code]["row_count"] += len(group)
        del df

    records = []
    for code, agg in hcpcs_agg.items():
        paid = agg["total_paid"]
        claims = agg["total_claims"]
        records.append({
            "hcpcs_code": code,
            "total_paid": paid,
            "total_claims": claims,
            "total_beneficiaries": agg["total_beneficiaries"],
            "avg_paid_per_claim": paid / claims if claims > 0 else 0,
            "n_providers": len(agg["provider_set"]),
            "n_years": len(agg["year_set"]),
            "row_count": agg["row_count"],
        })

    hcpcs_df = pd.DataFrame(records).sort_values("total_paid", ascending=False)

    output_path = root / config["paths"]["processed_dir"] / "hcpcs_features.parquet"
    hcpcs_df.to_parquet(output_path, index=False)
    logger.info(f"HCPCS features saved: {len(hcpcs_df):,} codes")

    return hcpcs_df


if __name__ == "__main__":
    import argparse
    from src.utils import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(description="Build derived features")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--providers", action="store_true", help="Build provider features")
    parser.add_argument("--timeseries", action="store_true", help="Build monthly time series")
    parser.add_argument("--hcpcs", action="store_true", help="Build HCPCS features")
    parser.add_argument("--all", action="store_true", help="Build all features")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.all or args.timeseries:
        build_monthly_time_series(config)
    if args.all or args.hcpcs:
        build_hcpcs_features(config)
    if args.all or args.providers:
        build_provider_features(config)
