#!/usr/bin/env python
"""Debug script to check column names from queries"""
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    config = Config()
    snowflake_config = config.get_snowflake_config()
    
    with SnowflakeConnector(snowflake_config) as conn:
        loader = DataLoader(conn)
        
        # Test price data
        print("=== Testing price data columns ===")
        price_data = loader.load_price_data_with_lookback(['AAPL'], '2024-01-15', 30)
        print(f"Shape: {price_data.shape}")
        print(f"Columns: {price_data.columns.tolist()}")
        print(f"First row: {price_data.iloc[0] if len(price_data) > 0 else 'No data'}")
        
        # Test market metrics
        print("\n=== Testing market metrics columns ===")
        market_data = loader.load_market_metrics(['AAPL'], '2024-01-15')
        print(f"Shape: {market_data.shape}")
        print(f"Columns: {market_data.columns.tolist()}")
        print(f"First row: {market_data.iloc[0] if len(market_data) > 0 else 'No data'}")


if __name__ == "__main__":
    main()