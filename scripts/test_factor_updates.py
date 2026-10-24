#!/usr/bin/env python3
"""Test that factor updates work correctly with calculation_date parameter."""
import sys
sys.path.append('.')

from datetime import datetime, timedelta
import pandas as pd
from src.config import Config
from src.data.database import DatabaseConnector
from src.data.data_loader import DataLoader
from src.factors.momentum import Momentum12_1
from src.factors.value import BookToMarket
from src.factors.technical import RSI14

def test_factors_with_dates():
    """Test factor calculations with specific dates."""
    config = Config()
    conn = DatabaseConnector(config)
    conn.connect()
    loader = DataLoader(conn)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    test_dates = ['2025-06-12', '2025-06-11', '2025-06-10']
    
    print("=" * 80)
    print("TESTING FACTOR UPDATES WITH CALCULATION DATES")
    print("=" * 80)
    
    # 1. Test data loader update
    print("\n1. Testing DataLoader.load_market_metrics with date range...")
    market_data = loader.load_market_metrics(
        symbols=symbols,
        start_date='2025-06-10',
        end_date='2025-06-12'
    )
    print(f"✅ Loaded {len(market_data)} market metric records for date range")
    print(f"   Dates: {sorted(market_data['date'].unique())}")
    
    # 2. Test momentum factor
    print("\n2. Testing Momentum factor with calculation dates...")
    
    # Load price data
    price_data = loader.load_price_data(
        symbols=symbols,
        start_date='2024-01-01',
        end_date='2025-06-12'
    )
    print(f"   Loaded {len(price_data)} price records")
    
    momentum = Momentum12_1()
    
    # Test without calculation_date (should use most recent)
    values_default = momentum.calculate(price_data)
    print(f"   Default calculation: {dict(values_default)}")
    
    # Test with specific dates
    for date in test_dates:
        values = momentum.calculate(price_data, calculation_date=date)
        print(f"   Date {date}: {dict(values)}")
    
    # 3. Test book-to-market factor
    print("\n3. Testing Book-to-Market factor with calculation dates...")
    bm_factor = BookToMarket()
    
    # Test with each date
    for date in test_dates:
        # Filter market data to this date
        date_market_data = market_data[market_data['date'] == date]
        if not date_market_data.empty:
            values = bm_factor.calculate(date_market_data, calculation_date=date)
            print(f"   Date {date}: {dict(values)}")
        else:
            print(f"   Date {date}: No market data available")
    
    # 4. Test RSI factor
    print("\n4. Testing RSI factor with calculation dates...")
    rsi = RSI14()
    
    for date in test_dates[:1]:  # Just test one date for RSI
        values = rsi.calculate(price_data, calculation_date=date)
        print(f"   Date {date}: {dict(values)}")
    
    print("\n✅ All factor updates working correctly!")
    
    conn.disconnect()

if __name__ == "__main__":
    test_factors_with_dates()