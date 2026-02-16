"""Data validation tests for the Medicaid Provider Spending dataset."""

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, get_project_root


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def project_root():
    return get_project_root()


class TestRawData:
    """Tests for raw data accessibility and schema."""

    def test_raw_file_exists(self, config, project_root):
        filepath = project_root / config["paths"]["raw_data"]
        assert filepath.exists(), f"Raw data file not found: {filepath}"

    def test_raw_file_not_empty(self, config, project_root):
        filepath = project_root / config["paths"]["raw_data"]
        assert filepath.stat().st_size > 0, "Raw data file is empty"

    def test_raw_schema(self, config, project_root):
        filepath = project_root / config["paths"]["raw_data"]
        df = pd.read_csv(filepath, nrows=10)
        expected_cols = {
            "BILLING_PROVIDER_NPI_NUM",
            "SERVICING_PROVIDER_NPI_NUM",
            "HCPCS_CODE",
            "CLAIM_FROM_MONTH",
            "TOTAL_UNIQUE_BENEFICIARIES",
            "TOTAL_CLAIMS",
            "TOTAL_PAID",
        }
        assert expected_cols == set(df.columns), f"Schema mismatch: {set(df.columns)}"


class TestProcessedData:
    """Tests for processed Parquet files."""

    def test_parquet_files_exist(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        parquet_files = list(processed_dir.glob("medicaid_*.parquet"))
        assert len(parquet_files) > 0, "No processed Parquet files found"

    def test_parquet_schema(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        for pf in processed_dir.glob("medicaid_*.parquet"):
            metadata = pq.read_metadata(pf)
            schema = pq.read_schema(pf)
            col_names = set(schema.names)
            required = {"BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE",
                        "CLAIM_FROM_MONTH", "TOTAL_CLAIMS", "TOTAL_PAID"}
            assert required.issubset(col_names), \
                f"{pf.name} missing columns: {required - col_names}"

    def test_no_null_billing_npi(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        for pf in processed_dir.glob("medicaid_*.parquet"):
            df = pd.read_parquet(pf, columns=["BILLING_PROVIDER_NPI_NUM"])
            null_count = df["BILLING_PROVIDER_NPI_NUM"].isna().sum()
            assert null_count == 0, \
                f"{pf.name} has {null_count} null BILLING_PROVIDER_NPI_NUM"

    def test_date_format(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        for pf in processed_dir.glob("medicaid_*.parquet"):
            df = pd.read_parquet(pf, columns=["CLAIM_FROM_MONTH"])
            sample = df["CLAIM_FROM_MONTH"].head(100)
            for val in sample:
                assert len(str(val)) == 7, f"Invalid date format: {val}"
                assert str(val)[4] == "-", f"Invalid date separator: {val}"

    def test_row_count_consistency(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        total_parquet_rows = 0
        for pf in processed_dir.glob("medicaid_*.parquet"):
            metadata = pq.read_metadata(pf)
            total_parquet_rows += metadata.num_rows

        # Should be close to raw count (allowing for potential cleaning)
        profiling_path = project_root / config["paths"]["tables_dir"] / "profiling_results.json"
        if profiling_path.exists():
            with open(profiling_path) as f:
                profile = json.load(f)
            raw_rows = profile["total_rows"]
            # Within 1% tolerance (accounting for cleaning)
            ratio = total_parquet_rows / raw_rows
            assert 0.99 <= ratio <= 1.01, \
                f"Row count mismatch: parquet={total_parquet_rows:,}, raw={raw_rows:,}"


class TestDerivedFeatures:
    """Tests for derived feature files."""

    def test_provider_features_exist(self, config, project_root):
        path = project_root / config["paths"]["processed_dir"] / "provider_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            assert len(df) > 0, "Provider features file is empty"
            assert "billing_npi" in df.columns
            assert "total_paid" in df.columns

    def test_monthly_time_series_exists(self, config, project_root):
        path = project_root / config["paths"]["processed_dir"] / "monthly_time_series.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            assert len(df) > 0, "Monthly time series is empty"
            assert "date" in df.columns or "CLAIM_FROM_MONTH" in df.columns

    def test_no_negative_claims(self, config, project_root):
        processed_dir = project_root / config["paths"]["processed_dir"]
        for pf in processed_dir.glob("medicaid_*.parquet"):
            df = pd.read_parquet(pf, columns=["TOTAL_CLAIMS"])
            neg = (df["TOTAL_CLAIMS"] < 0).sum()
            assert neg == 0, f"{pf.name} has {neg} negative claim counts"


class TestProfilingResults:
    """Tests for profiling output quality."""

    def test_profiling_json_exists(self, config, project_root):
        path = project_root / config["paths"]["tables_dir"] / "profiling_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            assert "total_rows" in data
            assert data["total_rows"] > 0

    def test_monthly_spending_csv(self, config, project_root):
        path = project_root / config["paths"]["tables_dir"] / "monthly_spending.csv"
        if path.exists():
            df = pd.read_csv(path)
            assert len(df) > 0
            assert "month" in df.columns
            assert "total_paid" in df.columns
