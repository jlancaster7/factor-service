#!/usr/bin/env python
"""Test the RSI technical factor with real data."""
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader
from src.factors.technical import RSI14
import pandas as pd
import numpy as np


def main():
    """Test RSI factor calculation with real data."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # logger = logging.getLogger(__name__)  # Not used
    
    # Initialize connection
    config = Config()
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    # Test companies
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    
    # Get recent data (need at least 15 days for 14-day RSI)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)  # Get 60 days to ensure enough data
    
    print("\n" + "="*60)
    print("TESTING RSI-14 TECHNICAL FACTOR")
    print("="*60)
    print(f"\nLoading data from {start_date} to {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    
    # Load price data
    query = f"""
    SELECT 
        c.symbol,
        d.date,
        p.adj_close,
        p.change_percent
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol IN ({','.join([f"'{s}'" for s in symbols])})
      AND d.date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY c.symbol, d.date
    """
    
    price_data = loader.query_to_dataframe(query)
    
    # Lowercase column names
    price_data.columns = price_data.columns.str.lower()
    
    print(f"\nLoaded {len(price_data)} price records")
    
    # Show recent price movements
    print("\nRecent price movements (last 5 days per symbol):")
    print("-" * 70)
    print(f"{'Symbol':<8} {'Date':<12} {'Price':<10} {'Change %':<10}")
    print("-" * 70)
    
    for symbol in symbols:
        symbol_data = price_data[price_data['symbol'] == symbol].tail(5)
        for _, row in symbol_data.iterrows():
            change_pct = float(row['change_percent']) if row['change_percent'] is not None else 0
            print(f"{row['symbol']:<8} {str(row['date']):<12} ${float(row['adj_close']):<9.2f} "
                  f"{change_pct:>9.2f}%")
        print()
    
    # Create and test factor
    factor = RSI14()
    
    print(f"\nCalculating {factor.name} factor...")
    print(f"Period: {factor.period} days")
    
    # Calculate with diagnostics
    values, diagnostics = factor.calculate_with_diagnostics(price_data)
    
    # Also calculate simple RSI for comparison
    simple_rsi = factor.calculate_simple(price_data)
    
    print("\n" + "-"*50)
    print("FACTOR VALUES")
    print("-"*50)
    print(f"{'Symbol':<8} {'RSI (Wilder)':<15} {'RSI (Simple)':<15} {'Signal':<15}")
    print("-"*50)
    
    for symbol in symbols:
        if symbol in values.index:
            rsi = values[symbol]
            simple = simple_rsi.get(symbol, np.nan)
            
            # Determine signal
            if rsi > 70:
                signal = "Overbought"
            elif rsi < 30:
                signal = "Oversold"
            elif rsi > 60:
                signal = "Strong"
            elif rsi < 40:
                signal = "Weak"
            else:
                signal = "Neutral"
            
            print(f"{symbol:<8} {rsi:<15.2f} {simple:<15.2f} {signal:<15}")
        else:
            print(f"{symbol:<8} {'NaN':<15} {'NaN':<15} {'No data':<15}")
    
    print("\n" + "-"*50)
    print("DIAGNOSTICS")
    print("-"*50)
    for key, value in diagnostics.items():
        if key != 'percentiles':
            print(f"{key}: {value}")
    
    if 'percentiles' in diagnostics and diagnostics['percentiles']['25%'] is not None:
        print("percentiles:")
        for pct, val in diagnostics['percentiles'].items():
            print(f"  {pct}: {val:.2f}")
    
    # Manual verification for one symbol
    print("\n" + "-"*50)
    print("MANUAL VERIFICATION (AAPL)")
    print("-"*50)
    
    aapl_data = price_data[price_data['symbol'] == 'AAPL'].sort_values('date')
    if len(aapl_data) > 14:
        # Get last 15 prices for manual calculation
        recent_prices = aapl_data['adj_close'].tail(15).values
        
        # Calculate changes
        changes = np.diff(recent_prices)
        
        # Show the changes
        print("Last 14 daily changes:")
        for i, change in enumerate(changes):
            gain = max(0, change)
            loss = max(0, -change)
            print(f"  Day {i+1}: {change:>7.2f} (Gain: {gain:>6.2f}, Loss: {loss:>6.2f})")
        
        # Calculate averages
        gains = changes[changes > 0]
        losses = -changes[changes < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        print(f"\nSimple average gain: {avg_gain:.4f}")
        print(f"Simple average loss: {avg_loss:.4f}")
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            manual_rsi = 100 - (100 / (1 + rs))
            print(f"RS = {rs:.4f}")
            print(f"Manual RSI (simple): {manual_rsi:.2f}")
            print(f"Factor RSI (Wilder): {values.get('AAPL', 'NaN')}")
    
    # Check market conditions
    print("\n" + "-"*50)
    print("MARKET CONDITIONS SUMMARY")
    print("-"*50)
    
    valid_values = values.dropna()
    if len(valid_values) > 0:
        overbought = sum(1 for v in valid_values if v > 70)
        oversold = sum(1 for v in valid_values if v < 30)
        neutral = sum(1 for v in valid_values if 40 <= v <= 60)
        
        print(f"Overbought (RSI > 70): {overbought} stocks")
        print(f"Oversold (RSI < 30): {oversold} stocks")
        print(f"Neutral (40-60): {neutral} stocks")
        print(f"Average RSI: {valid_values.mean():.2f}")
    
    # Reasonableness checks
    print("\n" + "-"*50)
    print("REASONABLENESS CHECKS")
    print("-"*50)
    
    if len(valid_values) > 0:
        print(f"✓ Calculated values for {len(valid_values)}/{len(symbols)} symbols")
        print(f"✓ All RSI values in [0, 100]? {all(0 <= v <= 100 for v in valid_values)}")
        print(f"✓ Any extreme values (>90 or <10)? {any(v > 90 or v < 10 for v in valid_values)}")
        print(f"✓ RSI distribution reasonable? Mean={valid_values.mean():.2f}, Std={valid_values.std():.2f}")
    else:
        print("⚠️ No valid values calculated!")
    
    connector.disconnect()
    print("\n✓ RSI factor test completed!")


if __name__ == "__main__":
    main()