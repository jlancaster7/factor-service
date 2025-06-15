#!/usr/bin/env python
"""Test the momentum factor with real data."""
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader
from src.factors.momentum import Momentum12_1
import pandas as pd


def main():
    """Test momentum factor calculation with real data."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize connection
    config = Config()
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    # Test companies
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    
    # Get data for the last 2 years to ensure we have enough history
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
    
    print("\n" + "="*60)
    print("TESTING MOMENTUM FACTOR")
    print("="*60)
    print(f"\nLoading data from {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Load price data
    query = f"""
    SELECT 
        c.symbol,
        d.date,
        p.adj_close
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol IN ({','.join([f"'{s}'" for s in symbols])})
      AND d.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY c.symbol, d.date
    """
    
    price_data = loader.query_to_dataframe(query)
    
    print(f"\nLoaded {len(price_data)} price records")
    # Lowercase column names for consistency
    price_data.columns = price_data.columns.str.lower()
    
    print(f"Date range: {price_data['date'].min()} to {price_data['date'].max()}")
    
    # Show data summary
    print("\nData summary by symbol:")
    summary = price_data.groupby('symbol').agg({
        'date': ['count', 'min', 'max'],
        'adj_close': ['min', 'max', 'mean']
    })
    print(summary)
    
    # Create and test factor
    factor = Momentum12_1()
    
    print(f"\n\nCalculating {factor.name} factor...")
    print(f"Lookback period: {factor.lookback_days} days (~12 months)")
    print(f"Skip period: {factor.skip_days} days (~1 month)")
    
    # Calculate with diagnostics
    values, diagnostics = factor.calculate_with_diagnostics(price_data)
    
    print("\n" + "-"*40)
    print("FACTOR VALUES")
    print("-"*40)
    for symbol in symbols:
        if symbol in values.index:
            print(f"{symbol}: {values[symbol]:.4f} ({values[symbol]*100:.2f}%)")
        else:
            print(f"{symbol}: NaN")
    
    print("\n" + "-"*40)
    print("DIAGNOSTICS")
    print("-"*40)
    for key, value in diagnostics.items():
        if key != 'percentiles':
            print(f"{key}: {value}")
    
    if 'percentiles' in diagnostics:
        print("percentiles:")
        for pct, val in diagnostics['percentiles'].items():
            print(f"  {pct}: {val}")
    
    # Manual verification for one symbol
    print("\n" + "-"*40)
    print("MANUAL VERIFICATION (AAPL)")
    print("-"*40)
    
    aapl_data = price_data[price_data['symbol'] == 'AAPL'].sort_values('date')
    if len(aapl_data) >= 252:
        current_price = aapl_data['adj_close'].iloc[-1]
        skip_price = aapl_data['adj_close'].iloc[-22]  # 21 days ago from most recent
        lookback_price = aapl_data['adj_close'].iloc[-253]  # 252 days ago from most recent
        
        manual_momentum = (skip_price / lookback_price) - 1
        
        print(f"Current price (most recent): ${current_price:.2f}")
        print(f"Skip price (21 days ago): ${skip_price:.2f}")
        print(f"Lookback price (252 days ago): ${lookback_price:.2f}")
        print(f"Manual calculation: ({skip_price:.2f} / {lookback_price:.2f}) - 1 = {manual_momentum:.4f}")
        print(f"Factor calculation: {values.get('AAPL', 'NaN')}")
        
        # Show the actual dates
        print(f"\nDates:")
        print(f"Current date: {aapl_data['date'].iloc[-1]}")
        print(f"Skip date: {aapl_data['date'].iloc[-22]}")
        print(f"Lookback date: {aapl_data['date'].iloc[-253]}")
    
    # Check for reasonableness
    print("\n" + "-"*40)
    print("REASONABLENESS CHECKS")
    print("-"*40)
    
    valid_values = values.dropna()
    if len(valid_values) > 0:
        print(f"✓ Calculated values for {len(valid_values)}/{len(symbols)} symbols")
        print(f"✓ All values between -100% and 500%? {all(-1 <= v <= 5 for v in valid_values)}")
        print(f"✓ Any extreme outliers (>300%)? {any(v > 3 for v in valid_values)}")
        print(f"✓ Any extreme negative (<-50%)? {any(v < -0.5 for v in valid_values)}")
    else:
        print("⚠️ No valid values calculated!")
    
    connector.disconnect()
    print("\n✓ Momentum factor test completed!")


if __name__ == "__main__":
    main()