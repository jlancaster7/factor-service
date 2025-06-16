# ETL Pipeline Implementation Plan for Factor Calculations

## Overview
Build a complete end-to-end ETL pipeline for our 3 existing factors (momentum_12_1, book_to_market, rsi_14) that handles extraction, transformation, and loading into our Snowflake factor tables.

## Architecture Approach

```
[Factor Calculations] → [Staging Layer] → [Analytics Layer] → [Monitoring/Logging]
                            ↓                     ↓                     ↓
                    STG_FACTOR_VALUES    FACT_DAILY_FACTORS    FACTOR_CALCULATION_LOG
```

## Implementation Components

### 1. Base ETL Framework
Leverage existing BaseETL from data service with adaptations

**BaseETL Review Findings:**
The provided BaseETL class from the data service is an excellent foundation with production-ready features:

✅ **What Works Well:**
- Solid architecture with clean abstract methods (extract, transform, load)
- Built-in retry logic with configurable attempts/delays
- Comprehensive batch processing (perfect for our bulk_insert operations)
- Detailed result tracking with ETLResult dataclass
- Production monitoring hooks and logging
- Performance tracking for each phase

🔧 **Adaptations Needed:**
1. **Remove External Dependencies:**
   - Remove FMPClient completely (not needed - we're reading from Snowflake)
   - Remove FMPTransformer (API-specific transformation logic)
   - Replace with FactorDataLoader and FactorRegistry
   - Set fmp_client=None in BaseETL constructor

2. **Constructor Pattern (from example):**
   ```python
   def __init__(self, config: Config):
       snowflake_connector = SnowflakeConnector(config.get_snowflake_config())
       super().__init__(
           job_name="factor_calculation_etl",
           snowflake_connector=snowflake_connector,
           fmp_client=None,  # Not needed
           batch_size=config.batch_size or 1000,
           enable_monitoring=False
       )
   ```

3. **Adapt Validation:**
   - Replace financial statement validation with factor-specific checks
   - Validate for NaN, Inf, outliers, reasonable ranges

4. **Monitoring Tables:**
   - Initially set `enable_monitoring=False` 
   - Add ETL monitoring tables in Phase 2

```python
# src/etl/base_etl.py
- Keep as-is from data service (already provided)

# src/etl/factor_etl.py  
- Create adapted base class for factor calculations
- Remove FMP dependencies
- Add factor-specific validation

```

### 2. Factor ETL Pipeline
Main ETL class for factor calculations

**Key Insights from MarketMetricsETL Example:**
Based on the provided example ETL, we should follow these patterns:

1. **Constructor Pattern:**
   ```python
   class FactorCalculationETL(BaseETL):
       def __init__(self, config: Config):
           snowflake_connector = SnowflakeConnector(config.get_snowflake_config())
           super().__init__(
               job_name="factor_calculation_etl",
               snowflake_connector=snowflake_connector,
               fmp_client=None,  # Not needed for factors
               batch_size=config.batch_size or 1000,
               enable_monitoring=False  # Initially disabled
           )
           self.config = config
   ```

2. **Extract Method:**
   - Use `snowflake.fetch_all(query, params)` to get data
   - Build queries with proper parameter binding
   - Store metadata in `self.result.metadata`

3. **Transform Method:**
   - Calculate factors using our FactorRegistry
   - Return custom dict like `{"staging": factor_values}`
   - Track errors and warnings during calculation

4. **Load Method:**
   - Add timestamps before loading: `record["calculation_timestamp"] = datetime.now(timezone.utc)`
   - Use `snowflake.bulk_insert("STAGING.STG_FACTOR_VALUES", data)`
   - Return count of records loaded

5. **Custom Run Method:**
   ```python
   def run(self, symbols=None, start_date=None, end_date=None):
       """Override base run() to accept specific parameters
       
       Args:
           symbols: List of symbols to process (None = all)
           start_date: Start date for calculation (supports backfill)
           end_date: End date for calculation (supports backfill)
       """
       # Support single date or date range
       if start_date and not end_date:
           end_date = start_date
           
       # Extract with parameters
       raw_data = self.extract(symbols, start_date, end_date)
       
       # Transform
       transformed_data = self.transform(raw_data)
       
       # Load
       records_loaded = self.load(transformed_data)
       
       return {
           "status": "success",
           "records_extracted": len(raw_data),
           "records_loaded": records_loaded,
           "errors": self.result.errors
       }
   ```

**Key Benefits from BaseETL:**
- Automatic retry on failures (configurable max_retries)
- Batch processing with configurable batch_size
- Comprehensive logging at each phase
- ETLResult tracking with duration metrics
- Error collection and reporting

### 3. Analytics Layer Processor
Process staging data into analytics fact table

```python
# src/etl/factor_analytics_processor.py
class FactorAnalyticsProcessor:
    def __init__(self, snowflake_connector: SnowflakeConnector):
        self.snowflake = snowflake_connector
        
    def process(self, calculation_date: str):
        # Read from staging using fetch_all()
        staging_data = self.snowflake.fetch_all(query, params)
        
        # Calculate cross-sectional statistics
        # Use SQL window functions for z-scores and percentiles
        
        # Use bulk_insert() to load FACT_DAILY_FACTORS
        self.snowflake.bulk_insert("ANALYTICS.FACT_DAILY_FACTORS", data)
```

**Key Operations:**
- Read staging data with proper joins to dimension tables
- Calculate statistics using SQL window functions (more efficient than Python)
- Map to dimension keys (company_key, date_key, factor_key)
- Bulk insert to analytics fact table

### 4. Factor ETL Runner Script
Orchestration script for the complete pipeline

```python
# scripts/run_factor_etl.py
- Orchestrate the complete pipeline
- Handle date parameters
- Log to FACTOR_CALCULATION_LOG
- Error handling and notifications
```

## Data Flow Details

### Step 1: Staging Layer Load
Write raw calculated values to staging using MERGE for idempotency

```python
# In the load() method
staging_data = transformed_data.get("staging", [])

# Add timestamps
current_timestamp = datetime.now(timezone.utc)
for record in staging_data:
    record["calculation_timestamp"] = current_timestamp

# Use MERGE for idempotent loads
merge_keys = ["symbol", "date", "factor_name"]
self.snowflake.merge(
    table="STAGING.STG_FACTOR_VALUES",
    data=staging_data,
    merge_keys=merge_keys,
    update_columns=["factor_value", "calculation_timestamp"]
)
```

### Step 2: Analytics Transform
Calculate statistics and load analytics layer

```python
# In FactorAnalyticsProcessor.process()
query = """
WITH factor_stats AS (
    SELECT 
        c.company_key,
        d.date_key,
        f.factor_key,
        stg.factor_value as raw_value,
        AVG(stg.factor_value) OVER (PARTITION BY stg.date, stg.factor_name) as daily_mean,
        STDDEV(stg.factor_value) OVER (PARTITION BY stg.date, stg.factor_name) as daily_std,
        PERCENT_RANK() OVER (PARTITION BY stg.date, stg.factor_name ORDER BY stg.factor_value) as percentile_rank
    FROM STAGING.STG_FACTOR_VALUES stg
    JOIN ANALYTICS.DIM_COMPANY c ON stg.symbol = c.symbol
    JOIN ANALYTICS.DIM_DATE d ON stg.date = d.date
    JOIN ANALYTICS.DIM_FACTOR f ON stg.factor_name = f.factor_name
    WHERE stg.date = %(calculation_date)s
)
SELECT 
    company_key,
    date_key,
    factor_key,
    raw_value,
    CASE 
        WHEN daily_std > 0 THEN (raw_value - daily_mean) / daily_std 
        ELSE 0 
    END as z_score,
    percentile_rank,
    NULL as sector_z_score,  -- Phase 2
    NULL as sector_percentile_rank,  -- Phase 2
    CURRENT_TIMESTAMP() as calculation_timestamp
FROM factor_stats
"""

# Fetch calculated statistics
analytics_data = self.snowflake.fetch_all(query, {"calculation_date": calculation_date})

# Convert to list of dicts for bulk_insert
data_to_load = [dict(row) for row in analytics_data]

# DELETE + INSERT pattern for analytics layer
with self.snowflake.transaction():
    # Delete existing data for this calculation date
    delete_query = """
    DELETE FROM ANALYTICS.FACT_DAILY_FACTORS 
    WHERE date_key IN (
        SELECT date_key FROM ANALYTICS.DIM_DATE WHERE date = %(calculation_date)s
    )
    """
    self.snowflake.execute(delete_query, {"calculation_date": calculation_date})
    
    # Bulk insert new calculated statistics
    self.snowflake.bulk_insert("ANALYTICS.FACT_DAILY_FACTORS", data_to_load)
```

### Step 3: Monitoring/Logging
Track calculation results and performance

```python
# Log calculation results to FACTOR_CALCULATION_LOG
log_entries = []
for factor_name, stats in factor_stats.items():
    log_entry = {
        'calculation_date': calculation_date,
        'factor_name': factor_name,
        'symbols_processed': stats['symbols_processed'],
        'records_calculated': stats['records_calculated'],
        'null_count': stats['null_count'],
        'inf_count': stats['inf_count'],
        'calculation_time_seconds': stats['elapsed_time'],
        'status': 'SUCCESS' if stats['errors'] == 0 else 'ERROR',
        'error_message': stats.get('error_message'),
        'created_timestamp': datetime.now(timezone.utc)
    }
    log_entries.append(log_entry)

# INSERT ONLY for monitoring logs (never update historical records)
self.snowflake.bulk_insert("ANALYTICS.FACTOR_CALCULATION_LOG", log_entries)
```

## Key Design Decisions

### 1. Two-Stage Process
- **Stage 1**: Raw values to staging (simple, fast)
- **Stage 2**: Statistical processing to analytics (complex, can be re-run)

### 2. Load Pattern by Layer
**Staging Layer (STG_FACTOR_VALUES)**: Use MERGE
- Idempotent operations - safe to re-run
- Preserves calculation_timestamp audit trail
- Handles late-arriving data gracefully
- Merge keys: [symbol, date, factor_name]

**Analytics Layer (FACT_DAILY_FACTORS)**: Use DELETE + INSERT
- Ensures statistical consistency (z-scores, percentiles)
- Replace all data for a calculation date atomically
- Simpler to implement and reason about
- Wrap in transaction for safety

**Monitoring Tables (FACTOR_CALCULATION_LOG)**: INSERT ONLY
- Never update historical logs
- Append-only for audit trail

### 3. Error Handling
- Continue on individual factor failures
- Log all errors but don't stop pipeline
- Return summary of successes/failures

### 4. Batch Processing
- Process all symbols for a date together
- Enables proper cross-sectional statistics
- Efficient bulk operations

## Implementation Order

### 1. Install Dependencies & Create Validation (~30 min) ✅ COMPLETE
- ✅ Install loguru: Already installed in venv and requirements.txt
- ✅ Create FactorDataValidator class with BaseETL-compatible interface
- ✅ Refactored to single file following KISS principle
- ✅ Tested with multiple data types (factor, non-factor, auto-detection)

### 2. Build FactorCalculationETL (~2 hours) ✅ COMPLETE
- ✅ Created factor_calculation_etl.py following the example pattern
- ✅ Implemented extract() to get price/market data using fetch_all()
  - Calculates lookback period based on factor requirements
  - Uses existing load_price_data() method (removed unnecessary lookback method)
  - Loads fundamentals and market metrics as needed
- ✅ Implemented transform() to calculate factors using FactorRegistry
  - Graceful error handling per factor
  - Comprehensive statistics collection
  - Successfully calculates book_to_market and rsi_14
- ✅ Implemented load() using MERGE to STG_FACTOR_VALUES
  - Idempotent loads with merge keys: [symbol, date, factor_name]
  - Adds calculation_timestamp to all records
- ✅ Override run() method with custom parameters
- ✅ Successfully tested with 5 companies - loaded 10 factor values
- ✅ Logs calculation results to FACTOR_CALCULATION_LOG

### 3. Build Analytics Processor (~2 hours)
- Calculate statistics
- Join with dimension tables
- Load to FACT_DAILY_FACTORS

### 4. Create Runner Script (~1 hour)
- Command line interface
- Date handling
- Logging integration

### 5. Testing & Validation (~1 hour)
- Test with 5 companies
- Verify statistics are correct
- Check idempotency

## Success Criteria

### 1. Complete Data Flow
- Factors calculate → staging → analytics → logs
- All 3 factors processing successfully

### 2. Data Quality
- Correct z-scores and percentiles
- No data loss or duplication
- Proper null handling

### 3. Operational Excellence
- Clear logging
- Error recovery
- Re-runnable for any date

### 4. Performance Baseline
- Establish timing for 5 companies
- Identify bottlenecks before scaling

## Design Decisions (Resolved)

### 1. Use MERGE Pattern ✓
- Use the snowflake_connector.merge() method for idempotent operations
- Merge keys: [symbol, date, factor_name] for STG_FACTOR_VALUES
- Safer than DELETE + INSERT, handles concurrent runs

### 2. Support Historical Backfill ✓
- Pipeline MUST support date ranges for adding new factors
- Example: `--start-date 2024-01-01 --end-date 2024-12-31`
- Essential for backfilling when new factors are added

### 3. Graceful Factor Failure Handling ✓
- Continue calculating other factors if one fails
- Log failures to FACTOR_CALCULATION_LOG table
- Monitoring schema coming from data service repo (hold off on implementation)

## Example Usage

```bash
# Calculate factors for today
python scripts/run_factor_etl.py

# Calculate factors for specific date
python scripts/run_factor_etl.py --date 2024-01-15

# Calculate factors for date range (if supported)
python scripts/run_factor_etl.py --start-date 2024-01-01 --end-date 2024-01-31

# Dry run mode
python scripts/run_factor_etl.py --date 2024-01-15 --dry-run
```

## Progress Update

### Completed:
- ✅ Step 1: Dependencies and validation setup complete
- ✅ BaseETL integrated and all import issues resolved
- ✅ FactorDataValidator refactored for direct BaseETL compatibility
- ✅ Step 2: FactorCalculationETL built and tested
- ✅ Staging layer tested with 3 factors (2 successful, 1 needs more data)

### Next Steps:
1. **Build Analytics Processor** (Step 3) - This is our immediate next task
2. Create runner script (Step 4)
3. Full testing and validation (Step 5)

## Dependencies Note

The BaseETL class has these dependencies we need to handle:
- ✅ `loguru` for logging - Installed and working
- ✅ ETLMonitor class - Handled with try/catch, sets `enable_monitoring=False` if not found
- ✅ DataQualityValidator - Using our FactorDataValidator with BaseETL-compatible interface
- ✅ FMPClient & FMPTransformer - Made optional in BaseETL constructor, set to None

All dependencies have been successfully handled!

## Key Patterns from Example

From the MarketMetricsETL example, we learned to:
1. **Use Config object** for initialization, not individual parameters
2. **Override run() method** to accept custom parameters (symbols, dates)
3. **Use fetch_all()** for queries with proper parameter binding
4. **Use bulk_insert()** for all data loading
5. **Add timestamps** before inserting data
6. **Return status dict** from run() method
7. **Store metadata** in self.result.metadata throughout execution