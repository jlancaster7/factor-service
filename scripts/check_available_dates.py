#!/usr/bin/env python
"""Check available dates for market metrics"""
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector


def main():
    config = Config()
    snowflake_config = config.get_snowflake_config()
    
    with SnowflakeConnector(snowflake_config) as conn:
        # Check latest available dates for market metrics
        query = """
        SELECT 
            c.symbol,
            MIN(d.date) as min_date,
            MAX(d.date) as max_date,
            COUNT(*) as record_count
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        WHERE c.symbol IN ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
        GROUP BY c.symbol
        ORDER BY c.symbol
        """
        
        print("=== Market Metrics Date Ranges ===")
        results = conn.fetch_all(query)
        for row in results:
            print(f"{row['SYMBOL']}: {row['MIN_DATE']} to {row['MAX_DATE']} ({row['RECORD_COUNT']} records)")
        
        # Check a specific date
        test_date = '2024-01-15'
        query2 = f"""
        SELECT COUNT(*) as count
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        WHERE d.date = '{test_date}'
        """
        
        result = conn.fetch_all(query2)
        print(f"\nRecords for {test_date}: {result[0]['COUNT'] if result else 0}")
        
        # Get most recent date with data
        query3 = """
        SELECT MAX(d.date) as latest_date
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        """
        
        result = conn.fetch_all(query3)
        print(f"Latest date with market metrics: {result[0]['LATEST_DATE'] if result else 'None'}")


if __name__ == "__main__":
    main()