#!/usr/bin/env python
"""Check all factor-related tables in the database."""
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    """List all factor-related tables."""
    # Load config
    config = Config()
    config.validate()
    
    # Connect to Snowflake
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    print("\n=== ALL FACTOR-RELATED TABLES ===\n")
    
    # Query for all tables with 'FACTOR' in the name
    query = """
    SELECT 
        TABLE_SCHEMA,
        TABLE_NAME,
        TABLE_TYPE,
        ROW_COUNT,
        CREATED
    FROM INFORMATION_SCHEMA.TABLES
    WHERE (TABLE_NAME LIKE '%FACTOR%' 
           OR TABLE_NAME = 'STG_FACTOR_VALUES')
    AND TABLE_SCHEMA IN ('ANALYTICS', 'STAGING')
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    
    tables = loader.query_to_dataframe(query)
    print(tables.to_string(index=False))
    
    # Check if each table is referenced in our current pipeline
    print("\n=== TABLE USAGE IN CURRENT PIPELINE ===\n")
    
    table_usage = {
        'ANALYTICS.DIM_FACTOR': 'Used by both ETLs for factor metadata',
        'ANALYTICS.FACT_FACTOR_RAW': 'Used by FactorCalculationETL (stores raw factor values)',
        'ANALYTICS.FACT_FACTOR_ANALYTICS': 'Used by FactorAnalyticsETL (stores z-scores and percentiles)',
        'ANALYTICS.FACTOR_CALCULATION_LOG': 'Used by FactorCalculationETL for logging and monitoring'
    }
    
    for table_name, usage in table_usage.items():
        print(f"{table_name:40} | {usage}")
    
    # Show record counts for each table
    print("\n=== RECORD COUNTS ===\n")
    
    for _, row in tables.iterrows():
        full_name = f"{row['table_schema']}.{row['table_name']}"
        count_query = f"SELECT COUNT(*) as cnt FROM {full_name}"
        try:
            result = connector.fetch_all(count_query)
            count = result[0]['CNT'] if result else 0
            print(f"{full_name:40} | {count:,} records")
        except Exception as e:
            print(f"{full_name:40} | Error: {str(e)}")
    
    
    # Check dependencies
    print("\n=== FACT_DAILY_FACTORS DEPENDENCIES ===\n")
    
    # Check if any views or procedures reference FACT_DAILY_FACTORS
    dep_query = """
    SELECT 
        REFERENCED_OBJECT_NAME,
        REFERENCING_OBJECT_TYPE,
        REFERENCING_OBJECT_NAME
    FROM SNOWFLAKE.ACCOUNT_USAGE.OBJECT_DEPENDENCIES
    WHERE REFERENCED_OBJECT_NAME = 'FACT_DAILY_FACTORS'
    AND REFERENCED_SCHEMA_NAME = 'ANALYTICS'
    """
    
    try:
        deps = loader.query_to_dataframe(dep_query)
        if len(deps) > 0:
            print("Dependencies found:")
            print(deps.to_string(index=False))
        else:
            print("No dependencies found on FACT_DAILY_FACTORS")
    except:
        print("Could not query object dependencies (may need ACCOUNT_USAGE access)")
    
    connector.disconnect()
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()