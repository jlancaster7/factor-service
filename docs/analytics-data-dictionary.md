# Analytics Layer Data Dictionary

## Overview

This document describes the data available in the ANALYTICS schema for equity factor calculation and portfolio construction. All data follows a star schema design with dimension and fact tables optimized for analytical queries.

## Table of Contents
1. [Dimension Tables](#dimension-tables)
2. [Fact Tables](#fact-tables)
3. [Data Coverage](#data-coverage)
4. [Common Query Patterns](#common-query-patterns)
5. [Factor Calculation Examples](#factor-calculation-examples)
6. [Data Quality Notes](#data-quality-notes)

## Dimension Tables

### DIM_COMPANY
Central dimension for all company information with SCD Type 2 history tracking.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| company_key | INTEGER | Surrogate key | 1 |
| symbol | VARCHAR(10) | Stock ticker symbol | AAPL |
| company_name | VARCHAR(255) | Full company name | Apple Inc. |
| sector | VARCHAR(100) | GICS sector classification | Technology |
| industry | VARCHAR(200) | GICS industry classification | Consumer Electronics |
| exchange | VARCHAR(50) | Primary exchange | NASDAQ |
| market_cap_category | VARCHAR(20) | Size classification | Mega Cap |
| headquarters_location | VARCHAR(500) | HQ city, state, country | Cupertino, CA, US |
| is_current | BOOLEAN | Current record flag | TRUE |
| valid_from | TIMESTAMP_NTZ | Record effective date | 2020-01-01 |
| valid_to | TIMESTAMP_NTZ | Record expiration date | 9999-12-31 |

**Market Cap Categories:**
- Mega Cap: >= $200B
- Large Cap: $10B - $200B  
- Mid Cap: $2B - $10B
- Small Cap: $300M - $2B
- Micro Cap: < $300M

### DIM_DATE
Pre-populated date dimension (2010-2030) with trading calendar information.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| date_key | INTEGER | Surrogate key (YYYYMMDD) | 20240115 |
| date | DATE | Calendar date | 2024-01-15 |
| year | INTEGER | Calendar year | 2024 |
| quarter | INTEGER | Calendar quarter (1-4) | 1 |
| month | INTEGER | Calendar month (1-12) | 1 |
| week | INTEGER | Week of year | 3 |
| day_of_month | INTEGER | Day of month | 15 |
| day_of_week | INTEGER | 1=Monday, 7=Sunday | 1 |
| day_name | VARCHAR(10) | Monday-Sunday | Monday |
| month_name | VARCHAR(10) | January-December | January |
| is_weekend | BOOLEAN | Saturday or Sunday | FALSE |
| is_month_end | BOOLEAN | Last day of month | FALSE |
| is_quarter_end | BOOLEAN | Last day of quarter | FALSE |
| is_year_end | BOOLEAN | Last day of year | FALSE |

## Fact Tables

### FACT_DAILY_PRICES
Daily price and volume data with calculated price changes.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| price_key | INTEGER | Surrogate key | 12345 |
| company_key | INTEGER | FK to DIM_COMPANY | 1 |
| date_key | INTEGER | FK to DIM_DATE | 20240115 |
| open_price | FLOAT | Opening price | 185.50 |
| high_price | FLOAT | Daily high | 187.25 |
| low_price | FLOAT | Daily low | 184.80 |
| close_price | FLOAT | Closing price | 186.90 |
| adj_close | FLOAT | Adjusted close (for splits) | 186.90 |
| volume | BIGINT | Shares traded | 45678900 |
| change_amount | FLOAT | Price change from previous day | 1.40 |
| change_percent | FLOAT | Percentage change | 0.75 |

**Key Points:**
- One record per company per trading day
- Adjusted close accounts for splits and dividends
- Change calculations use previous trading day (not calendar day)

### FACT_FINANCIALS
Quarterly and annual financial statement data with point-in-time information.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| financial_key | INTEGER | Surrogate key | 5678 |
| company_key | INTEGER | FK to DIM_COMPANY | 1 |
| fiscal_date_key | INTEGER | FK to DIM_DATE (period end) | 20230930 |
| filing_date_key | INTEGER | FK to DIM_DATE (SEC filing) | 20231030 |
| accepted_date | TIMESTAMP_NTZ | When data became public | 2023-10-30 16:05:00 |
| period_type | VARCHAR(10) | Q1/Q2/Q3/Q4/FY | Q3 |
| **Income Statement** | | | |
| revenue | FLOAT | Total revenue | 89497000000 |
| cost_of_revenue | FLOAT | Cost of goods sold | 45000000000 |
| gross_profit | FLOAT | Revenue - COGS | 44497000000 |
| operating_expenses | FLOAT | SG&A + R&D | 14000000000 |
| operating_income | FLOAT | EBIT | 30497000000 |
| net_income | FLOAT | Bottom line profit | 22956000000 |
| eps | FLOAT | Earnings per share | 1.46 |
| eps_diluted | FLOAT | Diluted EPS | 1.45 |
| shares_outstanding | BIGINT | Basic shares | 15728700000 |
| **Balance Sheet** | | | |
| total_assets | FLOAT | Total assets | 352755000000 |
| current_assets | FLOAT | Cash + receivables + inventory | 143692000000 |
| total_liabilities | FLOAT | Total liabilities | 274763000000 |
| current_liabilities | FLOAT | Due within 1 year | 145308000000 |
| total_equity | FLOAT | Shareholders' equity | 77992000000 |
| cash_and_equivalents | FLOAT | Cash + short-term investments | 29965000000 |
| total_debt | FLOAT | Short + long term debt | 109280000000 |
| net_debt | FLOAT | Total debt - cash | 79315000000 |
| **Cash Flow** | | | |
| operating_cash_flow | FLOAT | Cash from operations | 28723000000 |
| investing_cash_flow | FLOAT | Capex + investments | -10959000000 |
| financing_cash_flow | FLOAT | Debt + dividends + buybacks | -17975000000 |
| free_cash_flow | FLOAT | Operating - Capex | 25293000000 |
| capital_expenditures | FLOAT | Property & equipment | -3430000000 |
| dividends_paid | FLOAT | Cash dividends (negative) | -3805000000 |

**Key Points:**
- `accepted_date` is critical for point-in-time backtesting
- All values in reporting currency (usually USD)
- Negative values for cash outflows (capex, dividends)
- Quarterly (Q1-Q4) and annual (FY) data available

### FACT_FINANCIALS_TTM
Pre-calculated trailing twelve month metrics for efficiency.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| ttm_key | INTEGER | Surrogate key | 9012 |
| company_key | INTEGER | FK to DIM_COMPANY | 1 |
| calculation_date | DATE | Date TTM was calculated | 2024-01-15 |
| accepted_date | TIMESTAMP_NTZ | Latest filing date used | 2023-10-30 16:05:00 |
| quarters_included | INTEGER | Number of quarters (always 4) | 4 |
| **TTM Flow Metrics** | | Sum of 4 quarters | |
| ttm_revenue | FLOAT | 4-quarter revenue sum | 383285000000 |
| ttm_net_income | FLOAT | 4-quarter net income sum | 96995000000 |
| ttm_operating_income | FLOAT | 4-quarter EBIT sum | 114301000000 |
| ttm_eps_diluted | FLOAT | 4-quarter diluted EPS sum | 6.16 |
| ttm_operating_cash_flow | FLOAT | 4-quarter operating CF | 110543000000 |
| ttm_free_cash_flow | FLOAT | 4-quarter FCF | 99584000000 |
| ttm_capital_expenditures | FLOAT | 4-quarter capex | -10959000000 |
| ttm_dividends_paid | FLOAT | 4-quarter dividends | -15025000000 |
| **Point-in-Time Metrics** | | Latest quarter values | |
| latest_shares_outstanding | BIGINT | Most recent share count | 15552750000 |
| latest_total_assets | FLOAT | Most recent assets | 352755000000 |
| latest_total_equity | FLOAT | Most recent equity | 62146000000 |
| latest_total_debt | FLOAT | Most recent debt | 111088000000 |
| latest_cash_and_equivalents | FLOAT | Most recent cash | 29965000000 |
| latest_net_debt | FLOAT | Most recent net debt | 81123000000 |

**Key Points:**
- Updated whenever 4 quarters of data are available
- Respects point-in-time logic using accepted_date
- Flow metrics (revenue, income) are summed
- Stock metrics (assets, shares) use latest quarter

### FACT_FINANCIAL_RATIOS
Pre-calculated financial ratios for each reporting period.

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| ratio_key | INTEGER | Surrogate key | |
| financial_key | INTEGER | FK to FACT_FINANCIALS | |
| **Profitability Ratios** | | | |
| gross_margin | FLOAT | Gross profit margin % | (Gross Profit / Revenue) × 100 |
| operating_margin | FLOAT | Operating margin % | (Operating Income / Revenue) × 100 |
| net_margin | FLOAT | Net profit margin % | (Net Income / Revenue) × 100 |
| roe | FLOAT | Return on equity % | (Net Income / Avg Equity) × 100 |
| roa | FLOAT | Return on assets % | (Net Income / Avg Assets) × 100 |
| roic | FLOAT | Return on invested capital % | (NOPAT / Invested Capital) × 100 |
| **Efficiency Ratios** | | | |
| asset_turnover | FLOAT | Revenue / Assets | Revenue / Average Assets |
| inventory_turnover | FLOAT | COGS / Inventory | COGS / Average Inventory |
| receivables_turnover | FLOAT | Revenue / Receivables | Revenue / Average Receivables |
| **Leverage Ratios** | | | |
| debt_to_equity | FLOAT | Total Debt / Equity | Total Debt / Total Equity |
| debt_to_assets | FLOAT | Total Debt / Assets | Total Debt / Total Assets |
| interest_coverage | FLOAT | EBIT / Interest | Operating Income / Interest Expense |
| **Liquidity Ratios** | | | |
| current_ratio | FLOAT | Current Assets / Liabilities | Current Assets / Current Liabilities |
| quick_ratio | FLOAT | Quick Assets / Liabilities | (Current Assets - Inventory) / Current Liabilities |
| cash_ratio | FLOAT | Cash / Current Liabilities | Cash & Equivalents / Current Liabilities |
| **Per Share Metrics** | | | |
| book_value_per_share | FLOAT | Equity / Shares | Total Equity / Shares Outstanding |
| revenue_per_share | FLOAT | Revenue / Shares | Revenue / Shares Outstanding |
| cash_per_share | FLOAT | Cash / Shares | Cash & Equivalents / Shares Outstanding |
| fcf_per_share | FLOAT | FCF / Shares | Free Cash Flow / Shares Outstanding |

**Key Points:**
- Calculated for both quarterly and annual periods
- Some ratios use averages (beginning + ending) / 2
- NULL values when denominator is zero or negative

### FACT_MARKET_METRICS
Daily market-based valuation metrics combining price and fundamental data.

| Column | Type | Description | Formula |
|--------|------|-------------|---------|
| market_metrics_key | INTEGER | Surrogate key | |
| company_key | INTEGER | FK to DIM_COMPANY | |
| date_key | INTEGER | FK to DIM_DATE | |
| financial_key | INTEGER | FK to latest FACT_FINANCIALS | |
| close_price | FLOAT | Daily closing price | |
| fiscal_period | VARCHAR(3) | Period of financial data | Q3 |
| **Market Data** | | | |
| market_cap | FLOAT | Market capitalization | Price × Shares Outstanding |
| enterprise_value | FLOAT | EV | Market Cap + Debt - Cash |
| **Valuation Multiples** | | | |
| pe_ratio | FLOAT | Price/Earnings (quarterly) | Price / Quarterly EPS |
| pe_ratio_ttm | FLOAT | Price/Earnings (TTM) | Price / TTM EPS |
| pb_ratio | FLOAT | Price/Book | Price / Book Value per Share |
| ps_ratio | FLOAT | Price/Sales (quarterly) | Price / Revenue per Share |
| ps_ratio_ttm | FLOAT | Price/Sales (TTM) | Market Cap / TTM Revenue |
| ev_to_revenue | FLOAT | EV/Revenue (quarterly) | EV / Quarterly Revenue |
| ev_to_revenue_ttm | FLOAT | EV/Revenue (TTM) | EV / TTM Revenue |
| ev_to_ebitda | FLOAT | EV/EBITDA | EV / TTM Operating Income |
| ev_to_ebit | FLOAT | EV/EBIT | EV / TTM Operating Income |
| **Income Metrics** | | | |
| dividend_yield | FLOAT | Dividend yield % | (Dividends per Share / Price) × 100 |
| payout_ratio | FLOAT | Payout ratio % | (Dividends / Net Income) × 100 |
| peg_ratio | FLOAT | P/E to Growth | P/E / Earnings Growth Rate |

**Key Points:**
- One record per company per trading day
- Uses most recent financial data as of each date
- Point-in-time calculations prevent look-ahead bias
- NULL when financial data not yet available

## Data Coverage

### Current Data Loaded (as of 2025-06-13)

| Metric | Coverage |
|--------|----------|
| **Companies** | 5 (MSFT, AAPL, NVDA, AMZN, GOOGL) |
| **Date Range** | 2020-06-15 to 2025-06-12 |
| **Daily Prices** | 6,280 records (1,256 per company) |
| **Financial Statements** | 106 quarterly reports |
| **TTM Calculations** | 91 TTM periods |
| **Financial Ratios** | 101 ratio sets |
| **Market Metrics** | 6,280 daily calculations |

### Update Frequency
- **Daily Prices**: Updated daily after market close
- **Financials**: Updated within 24 hours of SEC filing
- **TTM/Ratios**: Recalculated after new financials
- **Market Metrics**: Calculated daily

## Common Query Patterns

### 1. Get Latest Financial Data
```sql
-- Most recent quarterly data for a company
SELECT f.*, c.symbol
FROM ANALYTICS.FACT_FINANCIALS f
JOIN ANALYTICS.DIM_COMPANY c ON f.company_key = c.company_key
WHERE c.symbol = 'AAPL'
  AND f.period_type IN ('Q1', 'Q2', 'Q3', 'Q4')
ORDER BY f.fiscal_date_key DESC
LIMIT 1;
```

### 2. Point-in-Time Data for Backtesting
```sql
-- Get financial data as it was known on a specific date
WITH point_in_time AS (
  SELECT 
    f.*,
    ROW_NUMBER() OVER (
      PARTITION BY f.company_key 
      ORDER BY f.accepted_date DESC
    ) as rn
  FROM ANALYTICS.FACT_FINANCIALS f
  WHERE f.accepted_date <= '2023-06-30'  -- As of date
    AND f.period_type IN ('Q1', 'Q2', 'Q3', 'Q4')
)
SELECT * FROM point_in_time WHERE rn = 1;
```

### 3. Time Series Analysis
```sql
-- Monthly average P/E ratios
SELECT 
  c.symbol,
  YEAR(d.date) as year,
  MONTH(d.date) as month,
  AVG(m.pe_ratio_ttm) as avg_pe_ttm,
  MIN(m.pe_ratio_ttm) as min_pe_ttm,
  MAX(m.pe_ratio_ttm) as max_pe_ttm
FROM ANALYTICS.FACT_MARKET_METRICS m
JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
WHERE m.pe_ratio_ttm IS NOT NULL
GROUP BY c.symbol, YEAR(d.date), MONTH(d.date)
ORDER BY c.symbol, year, month;
```


## Data Quality Notes

### 1. Point-in-Time Accuracy
- All financial data includes `accepted_date` for proper backtesting
- Market metrics use the most recent data available as of each date
- No look-ahead bias in calculations

### 2. Data Completeness
- Some companies may have missing quarters
- TTM calculations require 4 consecutive quarters
- Market metrics NULL when financial data unavailable

### 3. Adjustments
- Stock prices are split-adjusted (adj_close)
- Financial data is as-reported (no pro-forma adjustments)
- All amounts in reporting currency (usually USD)

### 4. Known Limitations
- No sector/industry adjustments
- No currency conversions for international companies
- Limited to quarterly reporting frequency

### 5. Best Practices for Factor Development
1. Always use `accepted_date` for point-in-time analysis
2. Handle NULL values appropriately (sector neutralization)
3. Consider market cap weighting for portfolio construction
4. Use TTM metrics for more stable factor values
5. Account for survivorship bias in backtests

## SQL Performance Tips

1. **Use Date Keys**: Join on integer date_key rather than date
2. **Partition Queries**: Use date ranges to limit data scanned
3. **Aggregate First**: Calculate metrics at company level before joining
4. **Index Usage**: Primary keys and foreign keys are indexed
5. **Window Functions**: Efficient for time-series calculations

## Contact

For questions about data availability or quality:
- Check `docs/operational-runbook.md` for troubleshooting
- Review ETL logs in monitoring tables
- Refer to `docs/ETL-COOKBOOK.md` for adding new data

---

*Last Updated: 2025-06-13*