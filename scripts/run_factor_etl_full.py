#!/usr/bin/env python3
"""Calculate local factors and analytics for an explicit or full source range."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.etl.factor_calculation_etl import FactorCalculationETL
from src.etl.factor_analytics_etl import FactorAnalyticsETL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date", help="YYYY-MM-DD; defaults to earliest price date"
    )
    parser.add_argument("--end-date", help="YYYY-MM-DD; defaults to latest price date")
    args = parser.parse_args()
    config = Config()
    with FactorCalculationETL(config) as calculation:
        coverage = calculation.database.fetch_one(
            """
            SELECT MIN(d.date) AS start_date, MAX(d.date) AS end_date
            FROM ANALYTICS.FACT_DAILY_PRICES p
            JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
        """
        )
        if not coverage["START_DATE"]:
            parser.exit(1, "No price data loaded. Use scripts/import_data.py first.\n")
        start = args.start_date or str(coverage["START_DATE"])
        end = args.end_date or str(coverage["END_DATE"])
        if start > end:
            parser.error("start-date must be on or before end-date")
        result = calculation.run(start_date=start, end_date=end)
        print(f"Raw factors: {result['records_loaded']:,}; status: {result['status']}")
        if result["errors"]:
            for error in result["errors"]:
                print(error, file=sys.stderr)
            return 1
        with FactorAnalyticsETL(config) as analytics:
            result = analytics.process_date_range(start, end)
            print(
                f"Analytics: {result['total_records']:,} rows across {result['dates_processed']} dates"
            )
            if result["failed"]:
                print(result["details"], file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
