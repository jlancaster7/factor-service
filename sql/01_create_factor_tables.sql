-- Create factor framework tables

-- Staging layer for calculated factors
CREATE TABLE IF NOT EXISTS STAGING.STG_FACTOR_VALUES (
    symbol VARCHAR(10),
    date DATE,
    factor_name VARCHAR(100),
    factor_value FLOAT,
    calculation_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (symbol, date, factor_name)
);

-- Analytics layer - Factor dimension
CREATE TABLE IF NOT EXISTS ANALYTICS.DIM_FACTOR (
    factor_key INTEGER IDENTITY(1,1),
    factor_name VARCHAR(100) UNIQUE NOT NULL,
    factor_category VARCHAR(50),
    factor_subcategory VARCHAR(50),
    calculation_frequency VARCHAR(20) DEFAULT 'daily',
    lookback_days INTEGER,
    description TEXT,
    formula TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (factor_key)
);

-- Analytics layer - Daily factor values
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_DAILY_FACTORS (
    factor_value_key INTEGER IDENTITY(1,1),
    company_key INTEGER,
    date_key INTEGER,
    factor_key INTEGER,
    raw_value FLOAT,
    z_score FLOAT,
    percentile_rank FLOAT,
    sector_z_score FLOAT,
    sector_percentile_rank FLOAT,
    calculation_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (factor_value_key),
    FOREIGN KEY (company_key) REFERENCES DIM_COMPANY(company_key),
    FOREIGN KEY (date_key) REFERENCES DIM_DATE(date_key),
    FOREIGN KEY (factor_key) REFERENCES DIM_FACTOR(factor_key),
    UNIQUE (company_key, date_key, factor_key)
);

-- Calculation log for monitoring
CREATE TABLE IF NOT EXISTS ANALYTICS.FACTOR_CALCULATION_LOG (
    log_id INTEGER IDENTITY(1,1),
    calculation_date DATE,
    factor_name VARCHAR(100),
    symbols_processed INTEGER,
    records_calculated INTEGER,
    null_count INTEGER,
    inf_count INTEGER,
    calculation_time_seconds FLOAT,
    status VARCHAR(20),
    error_message TEXT,
    created_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (log_id)
);

-- Insert our initial factors into DIM_FACTOR
MERGE INTO ANALYTICS.DIM_FACTOR AS target
USING (
    SELECT * FROM VALUES
        ('momentum_12_1', 'momentum', NULL, 'daily', 252, '12-month minus 1-month price momentum', '(Price_t-21 / Price_t-252) - 1'),
        ('book_to_market', 'value', NULL, 'daily', 0, 'Book value to market value ratio', 'Total Equity / (Price * Shares Outstanding)'),
        ('rsi_14', 'technical', NULL, 'daily', 14, '14-day Relative Strength Index', 'RSI = 100 - (100 / (1 + RS))')
) AS source (factor_name, factor_category, factor_subcategory, calculation_frequency, lookback_days, description, formula)
ON target.factor_name = source.factor_name
WHEN NOT MATCHED THEN
    INSERT (factor_name, factor_category, factor_subcategory, calculation_frequency, lookback_days, description, formula)
    VALUES (source.factor_name, source.factor_category, source.factor_subcategory, source.calculation_frequency, source.lookback_days, source.description, source.formula);