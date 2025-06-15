#!/usr/bin/env python
"""Test Snowflake connection."""
import sys
import logging

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    """Test connection to Snowflake."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = Config()
    config.validate()
    
    # Get Snowflake config for the new connector
    snowflake_config = config.get_snowflake_config()
    
    # Test connection
    print("\nTesting Snowflake connection...")
    connector = SnowflakeConnector(snowflake_config)
    
    # Test basic connection
    try:
        connector.connect()
        print("✓ Connection successful!")
        
        # Create data loader
        loader = DataLoader(connector)
        
        # Try a simple query
        print("\nTesting query execution...")
        result = loader.query_to_dataframe("SELECT CURRENT_TIMESTAMP() as current_time, CURRENT_DATABASE() as database")
        print("✓ Query successful!")
        print(f"\nResult:\n{result}")
        
        # Test schema access
        print(f"\nChecking tables in {config.snowflake_schema} schema...")
        tables_query = """
        SELECT TABLE_NAME, ROW_COUNT, BYTES
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = %(schema)s
        ORDER BY TABLE_NAME
        """
        tables = loader.query_to_dataframe(tables_query, {'schema': config.snowflake_schema})
        
        if len(tables) > 0:
            print(f"✓ Found {len(tables)} tables in {config.snowflake_schema} schema:")
            print(tables)
        else:
            print(f"No tables found in {config.snowflake_schema} schema (this is expected for a new setup)")
            
    except Exception as e:
        print(f"✗ Connection/Query failed: {e}")
        sys.exit(1)
    finally:
        connector.disconnect()
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()