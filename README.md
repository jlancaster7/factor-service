# Equity Factors Service

A local Python pipeline for calculating equity factors and cross-sectional
analytics. Storage is **DuckDB**, in a single file on your computer. There is no
Snowflake account, database server, or cloud warehouse bill required.

## What is implemented

- `momentum_12_1`: adjusted-price return from 252 trading observations ago to 21
  observations ago; needs at least 253 price observations.
- `book_to_market`: inverse of the supplied point-in-time price-to-book ratio.
- `rsi_14`: 14-day RSI with Wilder's smoothing.
- Raw factor upserts, market-wide z-scores and percentile ranks, calculation
  diagnostics, and audit logging.

Data flow:

```text
Local CSV/Parquet prices and market metrics
    → DuckDB source tables
    → FactorCalculationETL (Python/pandas)
    → ANALYTICS.FACT_FACTOR_RAW
    → FactorAnalyticsETL (DuckDB SQL)
    → ANALYTICS.FACT_FACTOR_ANALYTICS
```

## Setup

Python 3.9+ is required. The existing project environment uses Python 3.10.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python scripts/setup_database.py
python scripts/test_connection.py
```

The default database is `data/equity_factors.duckdb`, relative to the project
root, even when a script is launched from another directory. Override it with:

```bash
export DATABASE_PATH=/absolute/path/to/equity_factors.duckdb
```

`LOG_LEVEL` and `BATCH_SIZE` are also configurable. An existing `.env` containing
old Snowflake credentials does not affect the local backend. No credentials are
required and local calculation makes no API calls.

## Load source data

A new database starts empty. Creating local tables does **not** recover data
from the canceled Snowflake account. Use existing exports or reload the source
from your data provider.

Import CSV or Parquet files using symbol/date columns; company and date keys
are generated locally. Imports update matching symbol/date records, and each
file is committed atomically. Duplicate symbol/date rows within a file are
rejected.

Prices require `symbol,date,adj_close`:

```csv
symbol,date,adj_close,close_price,volume
AAPL,2025-06-10,202.67,202.67,50000000
MSFT,2025-06-10,470.92,470.92,20000000
```

These are format examples, not a market-data dataset. Optional columns are
`open_price,high_price,low_price,close_price,volume,change_amount,change_percent`,
plus `sector,company_name`. Always supply genuinely adjusted prices; the
importer does not infer adjusted prices from raw closes.

Market metrics require `symbol,date,pb_ratio`:

```csv
symbol,date,pb_ratio
AAPL,2025-06-10,45.0
MSFT,2025-06-10,12.0
```

Optional metrics are `close_price,market_cap,enterprise_value,pe_ratio,pe_ratio_ttm,
ps_ratio,ps_ratio_ttm,ev_to_revenue,ev_to_revenue_ttm,dividend_yield`, plus
`sector,company_name`. Historical P/B values must reflect information available
on each date. Current ratios must not be copied over historical dates.

```bash
python scripts/import_data.py prices /path/to/prices.csv
python scripts/import_data.py market-metrics /path/to/market_metrics.parquet
```

Parquet input requires a pandas Parquet engine, e.g. `pip install pyarrow`.
CSV input works with the core dependencies. The neighboring
`financial-data-service` repository now writes FMP data directly into this same
DuckDB file. Run its daily pipeline, close its connections, then run the factor
pipeline below. Both repositories must use the same absolute `DATABASE_PATH`
when overriding their defaults. See its README for annual/quarterly access,
API caching, and the daily request budget.

## Calculate factors and analytics

```bash
# Process the entire available price history:
python scripts/run_factor_etl_full.py

# Or process an explicit date range:
python scripts/run_factor_etl_full.py --start-date 2025-06-02 --end-date 2025-06-12
```

The runner reads the actual source-data date range; it does not default to
today. Calculations run only on dates and symbols with price observations.
Earlier history is loaded relative to the beginning of a requested backfill.
Insufficient price history produces no stored value for that factor. Without
market metrics, book-to-market is skipped; momentum and RSI can still run.
Cross-sectional analytics require at least two symbols with a value for a
factor/date. Sector statistics remain unimplemented.

Python usage:

```python
from src.config import Config
from src.etl.factor_calculation_etl import FactorCalculationETL
from src.etl.factor_analytics_etl import FactorAnalyticsETL

config = Config()
with FactorCalculationETL(config) as calculation:
    result = calculation.run(start_date="2025-06-02", end_date="2025-06-12")
with FactorAnalyticsETL(config) as analytics:
    summary = analytics.process_date_range("2025-06-02", "2025-06-12")
```

## Storage and operations

`src/data/schema.sql` creates the local source tables, factor definitions,
raw factors, analytics, and audit log. This source schema covers the current
factor pipeline; it is not a complete replica of the upstream financial data
warehouse. `sql/create_factor_tables.sql` is the historical Snowflake schema.

Run writes from one process at a time. Close ETLs/connections before opening
the file in another application for writing. Back up the database file after
all writing connections close. Local database files are ignored by Git.
There is no automated scheduling or remote backup configured.

`src/data/snowflake_connector.py` is a legacy import alias to the local
connector; it does not connect to Snowflake. Older notebooks and manual
scripts may still assume the former warehouse schema or configuration. The
supported entrypoints are setup, connection check, import, and full ETL runner
shown above. `src/etl/example_etl.py` is an old upstream reference, not a runnable
local pipeline.

## Development and verification

```bash
python -m pytest -q
black src/data/database.py src/data/local_import.py src/config.py tests scripts/setup_database.py scripts/test_connection.py scripts/test_config.py scripts/import_data.py scripts/run_factor_etl_full.py
```

The offline integration tests use temporary DuckDB files and synthetic data.
They cover persistence, repeatable imports/calculations, both ETL stages,
backfill history, dates without prices, transaction rollback, and command-line
execution. Pytest is restricted to `tests/`; manual scripts under `scripts/`
are not collected as automated tests. Older planning documents describe
historical Snowflake architecture; this README describes the current backend.
