CREATE SCHEMA IF NOT EXISTS ANALYTICS;
CREATE SEQUENCE IF NOT EXISTS ANALYTICS.company_seq;
CREATE SEQUENCE IF NOT EXISTS ANALYTICS.factor_seq;
CREATE SEQUENCE IF NOT EXISTS ANALYTICS.log_seq;

CREATE TABLE IF NOT EXISTS ANALYTICS.DIM_COMPANY (
    company_key BIGINT PRIMARY KEY DEFAULT nextval('ANALYTICS.company_seq'),
    symbol VARCHAR UNIQUE NOT NULL,
    company_name VARCHAR,
    sector VARCHAR,
    is_current BOOLEAN DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS ANALYTICS.DIM_DATE (
    date_key INTEGER PRIMARY KEY,
    date DATE UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_DAILY_PRICES (
    company_key BIGINT NOT NULL,
    date_key INTEGER NOT NULL,
    open_price DOUBLE, high_price DOUBLE, low_price DOUBLE,
    close_price DOUBLE, adj_close DOUBLE, volume BIGINT,
    change_amount DOUBLE, change_percent DOUBLE,
    PRIMARY KEY (company_key, date_key)
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_FINANCIALS (
    company_key BIGINT NOT NULL,
    fiscal_date_key INTEGER NOT NULL,
    accepted_date TIMESTAMP NOT NULL,
    total_equity DOUBLE,
    shares_outstanding DOUBLE,
    PRIMARY KEY (company_key, fiscal_date_key, accepted_date)
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_MARKET_METRICS (
    company_key BIGINT NOT NULL,
    date_key INTEGER NOT NULL,
    close_price DOUBLE, market_cap DOUBLE, enterprise_value DOUBLE,
    pe_ratio DOUBLE, pe_ratio_ttm DOUBLE, pb_ratio DOUBLE,
    ps_ratio DOUBLE, ps_ratio_ttm DOUBLE,
    ev_to_revenue DOUBLE, ev_to_revenue_ttm DOUBLE, dividend_yield DOUBLE,
    PRIMARY KEY (company_key, date_key)
);
CREATE TABLE IF NOT EXISTS ANALYTICS.DIM_FACTOR (
    factor_key BIGINT PRIMARY KEY DEFAULT nextval('ANALYTICS.factor_seq'),
    factor_name VARCHAR UNIQUE NOT NULL,
    factor_category VARCHAR,
    factor_subcategory VARCHAR,
    calculation_frequency VARCHAR DEFAULT 'daily',
    lookback_days INTEGER,
    description VARCHAR, formula VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_FACTOR_RAW (
    company_key BIGINT NOT NULL,
    date_key INTEGER NOT NULL,
    factor_key BIGINT NOT NULL,
    factor_value DOUBLE,
    calculation_timestamp TIMESTAMP,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_key, date_key, factor_key)
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_FACTOR_ANALYTICS (
    company_key BIGINT NOT NULL,
    date_key INTEGER NOT NULL,
    factor_key BIGINT NOT NULL,
    raw_value DOUBLE, z_score DOUBLE, percentile_rank DOUBLE,
    sector_z_score DOUBLE, sector_percentile_rank DOUBLE,
    calculation_timestamp TIMESTAMP,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (company_key, date_key, factor_key)
);
CREATE TABLE IF NOT EXISTS ANALYTICS.FACTOR_CALCULATION_LOG (
    log_id BIGINT PRIMARY KEY DEFAULT nextval('ANALYTICS.log_seq'),
    calculation_date DATE, factor_name VARCHAR,
    symbols_processed INTEGER, records_calculated INTEGER,
    null_count INTEGER, inf_count INTEGER,
    calculation_time_seconds DOUBLE,
    status VARCHAR, error_message VARCHAR,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO ANALYTICS.DIM_FACTOR
    (factor_name, factor_category, lookback_days, description, formula)
VALUES
    ('momentum_12_1', 'momentum', 252, '12-month minus 1-month momentum', '(Price_t-21 / Price_t-252) - 1'),
    ('book_to_market', 'value', 0, 'Inverse price-to-book', '1 / pb_ratio'),
    ('rsi_14', 'technical', 14, '14-day RSI', '100 - (100 / (1 + RS))')
ON CONFLICT (factor_name) DO NOTHING;
