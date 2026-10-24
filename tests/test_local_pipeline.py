"""Offline integration tests against a real, temporary DuckDB file."""

from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data.database import DatabaseConnector
from src.data.local_import import import_frame
from src.etl.factor_calculation_etl import FactorCalculationETL
from src.etl.factor_analytics_etl import FactorAnalyticsETL


@pytest.fixture
def config(tmp_path):
    return Config(database_path=str(tmp_path / "test.duckdb"))


def seed(config):
    dates = pd.bdate_range("2023-01-02", periods=300)
    frames = []
    for i, symbol in enumerate(["AAA", "BBB", "CCC"]):
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "adj_close": 100 + np.arange(len(dates)) * (i + 1),
                    "close_price": 100 + np.arange(len(dates)) * (i + 1),
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    metrics = prices[["symbol", "date"]].copy()
    metrics["pb_ratio"] = metrics["symbol"].map({"AAA": 2.0, "BBB": 4.0, "CCC": 5.0})
    with DatabaseConnector(config) as db:
        import_frame(db, prices, "prices")
        import_frame(db, metrics, "market-metrics")
    return dates


def test_full_pipeline_is_persistent_and_idempotent(config):
    dates = seed(config)
    start, end = str(dates[-8].date()), str(dates[-1].date())
    with FactorCalculationETL(config) as calculation:
        first = calculation.run(start_date=start, end_date=end)
        assert first["status"] == "SUCCESS"
        assert first["records_loaded"] == 8 * 3 * 3
        second = calculation.run(start_date=start, end_date=end)
        assert second["records_loaded"] == first["records_loaded"]
    with FactorAnalyticsETL(config) as analytics:
        for _ in range(2):
            result = analytics.process_date_range(start, end)
            assert result["successful"] == 8
            assert result["total_records"] == 72
    # Reopen from disk and inspect actual outputs, not only status counters.
    with DatabaseConnector(config) as db:
        assert (
            db.fetch_one("SELECT COUNT(*) AS n FROM ANALYTICS.FACT_FACTOR_RAW")["N"]
            == 72
        )
        assert (
            db.fetch_one("SELECT COUNT(*) AS n FROM ANALYTICS.FACT_FACTOR_ANALYTICS")[
                "N"
            ]
            == 72
        )
        stats = db.fetch_all(
            """
            SELECT date_key, factor_key, AVG(z_score) AS mean_z,
                   MIN(percentile_rank) AS low, MAX(percentile_rank) AS high
            FROM ANALYTICS.FACT_FACTOR_ANALYTICS GROUP BY date_key, factor_key
        """
        )
        assert len(stats) == 24
        for row in stats:
            assert row["MEAN_Z"] == pytest.approx(0, abs=1e-12)
            assert row["LOW"] == 0
            assert 0 <= row["HIGH"] <= 1
        bm = db.fetch_one(
            """
            SELECT r.factor_value FROM ANALYTICS.FACT_FACTOR_RAW r
            JOIN ANALYTICS.DIM_FACTOR f USING (factor_key)
            JOIN ANALYTICS.DIM_COMPANY c USING (company_key)
            WHERE f.factor_name = 'book_to_market' AND c.symbol = 'AAA' LIMIT 1
        """
        )
        assert bm["FACTOR_VALUE"] == 0.5


def test_backfill_loads_history_for_start_and_ignores_non_price_dates(config):
    dates = seed(config)
    start, end = str(dates[260].date()), str(dates[-1].date())
    # A date dimension can contain weekends/future dates with no observations.
    future = dates[-1].date() + timedelta(days=3)
    with DatabaseConnector(config) as db:
        db.bulk_insert(
            "ANALYTICS.DIM_DATE",
            [{"date_key": int(future.strftime("%Y%m%d")), "date": future}],
        )
    with FactorCalculationETL(config) as calculation:
        result = calculation.run(start_date=start, end_date=str(future))
        assert result["records_loaded"] == 40 * 3 * 3
        assert result["factor_stats"]["momentum_12_1"]["total_nulls"] == 0
        assert calculation.calculation_dates[-1] == end


def test_import_updates_without_duplicates_and_rejects_bad_prices(config):
    prices = pd.DataFrame(
        {"symbol": ["AAA"], "date": ["2025-01-02"], "adj_close": [100.0]}
    )
    with DatabaseConnector(config) as db:
        import_frame(db, prices, "prices")
        prices["adj_close"] = 120.0
        import_frame(db, prices, "prices")
        assert (
            db.fetch_one("SELECT adj_close FROM ANALYTICS.FACT_DAILY_PRICES")[
                "ADJ_CLOSE"
            ]
            == 120.0
        )
        assert db.fetch_one("SELECT COUNT(*) AS n FROM ANALYTICS.DIM_COMPANY")["N"] == 1
        with pytest.raises(ValueError, match="Duplicate"):
            import_frame(db, pd.concat([prices, prices]), "prices")
        with pytest.raises(ValueError, match="Missing required"):
            import_frame(
                db, prices.rename(columns={"adj_close": "close_price"}), "prices"
            )


def test_analytics_rolls_back_failed_replacement(config, monkeypatch):
    dates = seed(config)
    day = str(dates[-1].date())
    with FactorCalculationETL(config) as calculation:
        calculation.run(start_date=day)
    with FactorAnalyticsETL(config) as analytics:
        assert analytics.run(day)["records_loaded"] == 9
        original = analytics.database.bulk_insert

        def fail(*args, **kwargs):
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(analytics.database, "bulk_insert", fail)
        with pytest.raises(RuntimeError, match="simulated"):
            analytics.run(day)
        assert (
            analytics.database.fetch_one(
                "SELECT COUNT(*) AS n FROM ANALYTICS.FACT_FACTOR_ANALYTICS"
            )["N"]
            == 9
        )
        monkeypatch.setattr(analytics.database, "bulk_insert", original)
        assert analytics.run(day)["status"] == "SUCCESS"


def test_no_snowflake_credentials_needed_and_empty_database_is_clear(
    config, monkeypatch
):
    for name in ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]:
        monkeypatch.delenv(name, raising=False)
    config.validate()
    with FactorCalculationETL(config) as calculation:
        with pytest.raises(ValueError, match="No companies loaded"):
            calculation.run()
    with FactorAnalyticsETL(config) as analytics:
        assert analytics.run("2025-01-02")["records_loaded"] == 0


def test_command_line_runner_works_outside_project_directory(config, tmp_path):
    dates = seed(config)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_factor_etl_full.py"
    result = subprocess.run(
        [sys.executable, str(script), "--start-date", str(dates[-2].date())],
        cwd=tmp_path,
        env={**os.environ, "DATABASE_PATH": config.database_path},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Raw factors: 18; status: SUCCESS" in result.stdout
    assert "Analytics: 18 rows across 2 dates" in result.stdout
