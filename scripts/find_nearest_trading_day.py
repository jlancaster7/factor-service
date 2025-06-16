#!/usr/bin/env python
"""Find nearest trading day to a given date"""
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector


def main():
    config = Config()
    snowflake_config = config.get_snowflake_config()
    
    with SnowflakeConnector(snowflake_config) as conn:
        # Check if 2024-01-15 is a trading day
        query = """
        SELECT 
            d.date,
            d.day_name,
            d.is_weekend,
            CASE WHEN m.date_key IS NOT NULL THEN 'YES' ELSE 'NO' END as has_market_data
        FROM ANALYTICS.DIM_DATE d
        LEFT JOIN (
            SELECT DISTINCT date_key 
            FROM ANALYTICS.FACT_MARKET_METRICS
        ) m ON d.date_key = m.date_key
        WHERE d.date BETWEEN '2024-01-10' AND '2024-01-20'
        ORDER BY d.date
        """
        
        print("=== Trading Days Around 2024-01-15 ===")
        results = conn.fetch_all(query)
        for row in results:
            weekend_flag = " (WEEKEND)" if row['IS_WEEKEND'] else ""
            print(f"{row['DATE']}: {row['DAY_NAME']}{weekend_flag} - Market Data: {row['HAS_MARKET_DATA']}")
        
        # Find the nearest trading day with data
        query2 = """
        SELECT d.date
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        WHERE d.date <= '2024-01-15'
        ORDER BY d.date DESC
        LIMIT 1
        """
        
        result = conn.fetch_all(query2)
        if result:
            print(f"\nNearest trading day on or before 2024-01-15: {result[0]['DATE']}")


if __name__ == "__main__":
    main()