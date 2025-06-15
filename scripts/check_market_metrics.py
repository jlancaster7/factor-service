#!/usr/bin/env python
"""Check what's available in the FACT_MARKET_METRICS table."""
import sys
import logging

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    """Check market metrics data availability."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize connection
    config = Config()
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    print("\n" + "="*60)
    print("CHECKING FACT_MARKET_METRICS TABLE")
    print("="*60)
    
    # Check table structure
    schema_query = """
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'ANALYTICS'
      AND TABLE_NAME = 'FACT_MARKET_METRICS'
    ORDER BY ORDINAL_POSITION
    """
    
    columns = loader.query_to_dataframe(schema_query)
    print("\nTable columns:")
    print(columns.to_string(index=False))
    
    # Check sample data
    print("\n" + "-"*60)
    print("SAMPLE DATA")
    print("-"*60)
    
    sample_query = """
    SELECT 
        c.symbol,
        d.date,
        m.pb_ratio,
        m.pe_ratio,
        m.ps_ratio,
        m.ev_to_ebitda,
        m.close_price,
        m.market_cap
    FROM ANALYTICS.FACT_MARKET_METRICS m
    JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
    WHERE c.symbol IN ('AAPL', 'MSFT', 'NVDA')
      AND d.date >= DATEADD('day', -30, CURRENT_DATE())
    ORDER BY c.symbol, d.date DESC
    LIMIT 15
    """
    
    sample_data = loader.query_to_dataframe(sample_query)
    if len(sample_data) > 0:
        sample_data.columns = sample_data.columns.str.lower()
        print(sample_data.to_string(index=False))
    
    # Check data availability for our test companies
    print("\n" + "-"*60)
    print("DATA AVAILABILITY FOR TEST COMPANIES")
    print("-"*60)
    
    availability_query = """
    SELECT 
        c.symbol,
        COUNT(*) as record_count,
        MIN(d.date) as min_date,
        MAX(d.date) as max_date,
        SUM(CASE WHEN m.pb_ratio IS NULL THEN 1 ELSE 0 END) as null_pb_count,
        AVG(m.pb_ratio) as avg_pb_ratio,
        MIN(m.pb_ratio) as min_pb_ratio,
        MAX(m.pb_ratio) as max_pb_ratio
    FROM ANALYTICS.FACT_MARKET_METRICS m
    JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
    WHERE c.symbol IN ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    availability = loader.query_to_dataframe(availability_query)
    if len(availability) > 0:
        availability.columns = availability.columns.str.lower()
        print(availability.to_string(index=False))
    
    connector.disconnect()
    print("\n✓ Market metrics check completed!")


if __name__ == "__main__":
    main()