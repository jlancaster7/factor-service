#!/usr/bin/env python
"""Set up required database tables for factor framework."""
import sys
import logging
from pathlib import Path

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    """Create factor framework tables in Snowflake."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load config
    config = Config()
    config.validate()
    
    # Connect to Snowflake
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    # Read SQL file
    sql_file = Path('sql/01_create_factor_tables.sql')
    logger.info(f"Reading SQL from {sql_file}")
    
    with open(sql_file, 'r') as f:
        sql_content = f.read()
    
    # Split by semicolons and execute each statement
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    logger.info(f"Found {len(statements)} SQL statements to execute")
    
    success_count = 0
    for i, statement in enumerate(statements, 1):
        logger.info(f"\nExecuting statement {i}/{len(statements)}:")
        logger.info(f"{statement[:100]}...")  # Show first 100 chars
        
        try:
            connector.execute(statement)
            logger.info("✓ Success")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed: {e}")
            # Continue with other statements
    
    logger.info(f"\nCompleted: {success_count}/{len(statements)} statements executed successfully")
    
    # Verify tables were created
    logger.info("\nVerifying table creation...")
    
    # Check ANALYTICS schema tables
    analytics_query = """
    SELECT TABLE_NAME, ROW_COUNT
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'ANALYTICS'
    AND TABLE_NAME IN ('DIM_FACTOR', 'FACT_DAILY_FACTORS', 'FACTOR_CALCULATION_LOG')
    ORDER BY TABLE_NAME
    """
    
    analytics_tables = loader.query_to_dataframe(analytics_query)
    logger.info(f"\nAnalytics tables:\n{analytics_tables}")
    
    # Check STAGING schema tables
    staging_query = """
    SELECT TABLE_NAME, ROW_COUNT
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'STAGING'
    AND TABLE_NAME = 'STG_FACTOR_VALUES'
    """
    
    staging_tables = loader.query_to_dataframe(staging_query)
    logger.info(f"\nStaging tables:\n{staging_tables}")
    
    # Check factor dimension entries
    factor_query = "SELECT factor_name, factor_category, lookback_days FROM ANALYTICS.DIM_FACTOR ORDER BY factor_name"
    factors = loader.query_to_dataframe(factor_query)
    logger.info(f"\nRegistered factors:\n{factors}")
    
    connector.disconnect()
    logger.info("\n✓ Database setup complete!")


if __name__ == "__main__":
    main()