# Analytics Processor Design (As Implemented)

## Purpose
Transform raw factor values from FACT_FACTOR_RAW into standardized analytics-ready data with cross-sectional statistics in FACT_FACTOR_ANALYTICS.

## Design Evolution
**Original Plan**: Simple class without BaseETL
**Actual Implementation**: FactorAnalyticsETL inheriting from BaseETL

## Why We Changed the Design

1. **Consistency**: Both ETLs (calculation and analytics) now follow the same pattern
2. **Reliability**: BaseETL provides retry logic, error handling, and monitoring
3. **Maintainability**: Single pattern for all ETL processes
4. **Future-Proofing**: Easy to add more complex transformations later

## Data Flow
```
FACT_FACTOR_RAW → Calculate Statistics → FACT_FACTOR_ANALYTICS
    (raw)        (z-scores, percentiles)      (analytics)
```

## Implementation Details

### 1. Class Structure
```python
class FactorAnalyticsETL(BaseETL):
    def __init__(self, config: Config):
        snowflake_connector = SnowflakeConnector(config.get_snowflake_config())
        super().__init__(
            job_name="factor_analytics_etl",
            snowflake_connector=snowflake_connector,
            fmp_client=None,
            batch_size=config.batch_size or 1000,
            enable_monitoring=False
        )
```

### 2. Statistics Calculation
Uses SQL window functions for efficiency:
- **MEAN/STDDEV**: Calculate per factor per date
- **Z-Score**: (value - mean) / stddev
- **Percentile Rank**: PERCENT_RANK() window function
- **Sector Stats**: Placeholders for future enhancement

### 3. Transaction Management
```python
# Manual transaction handling (SnowflakeConnector doesn't have context manager)
self.snowflake.execute("BEGIN")
try:
    # DELETE existing analytics
    self.snowflake.execute(delete_query, params)
    # INSERT new analytics
    self.snowflake.bulk_insert("ANALYTICS.FACT_FACTOR_ANALYTICS", data)
    # COMMIT
    self.snowflake.execute("COMMIT")
except Exception:
    self.snowflake.execute("ROLLBACK")
    raise
```

## Key Features Implemented

### 1. Extract Method
- Reads from FACT_FACTOR_RAW for a specific date
- Joins with dimension tables for context
- Validates data exists before processing

### 2. Transform Method  
- Single SQL query with CTEs for all statistics
- Window functions partition by date_key and factor_key
- Handles edge cases (zero std dev)
- Tracks statistics by factor for monitoring

### 3. Load Method
- DELETE + INSERT pattern for atomicity
- Ensures consistent statistics for a date
- Transaction wrapping for data integrity

### 4. Run Method
- Accepts calculation_date parameter
- Returns detailed status and statistics
- Supports date range processing

## SQL Strategy

```sql
WITH factor_stats AS (
    SELECT 
        fr.company_key,
        fr.date_key,
        fr.factor_key,
        fr.factor_value as raw_value,
        AVG(fr.factor_value) OVER (PARTITION BY fr.date_key, fr.factor_key) as factor_mean,
        STDDEV(fr.factor_value) OVER (PARTITION BY fr.date_key, fr.factor_key) as factor_std,
        PERCENT_RANK() OVER (
            PARTITION BY fr.date_key, fr.factor_key 
            ORDER BY fr.factor_value
        ) as percentile_rank
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    WHERE fr.date_key = (SELECT date_key FROM DIM_DATE WHERE date = ?)
)
SELECT 
    company_key,
    date_key,
    factor_key,
    raw_value,
    CASE 
        WHEN factor_std > 0 THEN (raw_value - factor_mean) / factor_std
        ELSE 0
    END as z_score,
    percentile_rank,
    NULL as sector_z_score,
    NULL as sector_percentile_rank
FROM factor_stats
```

## Testing Results

Successfully tested with:
- 5 companies × 3 factors = 15 records
- Z-scores properly normalized (mean ~0, std ~1)
- Percentiles correctly distributed [0, 1]
- All raw records processed without loss

## Benefits of Final Design

1. **Consistency**: Same pattern as FactorCalculationETL
2. **Reliability**: Inherits BaseETL's error handling and retry logic
3. **Performance**: SQL-based calculation is very efficient
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add sector statistics later

## Future Enhancements

1. **Sector Statistics**: Calculate sector-relative metrics
2. **Industry Groups**: Add industry-level statistics
3. **Time-Series Stats**: Rolling averages and volatility
4. **Outlier Detection**: Flag extreme values
5. **Data Quality Metrics**: Track coverage and completeness