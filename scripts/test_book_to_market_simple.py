#!/usr/bin/env python
"""Test the book-to-market factor using P/B ratios from market metrics."""
import sys
import logging

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader
from src.factors.value import BookToMarket


def main():
    """Test book-to-market factor calculation."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize connection
    config = Config()
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    # Test companies
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    
    print("\n" + "="*60)
    print("TESTING BOOK-TO-MARKET FACTOR (using P/B ratios)")
    print("="*60)
    print(f"\nSymbols: {', '.join(symbols)}")
    
    # Get the most recent P/B ratios from market metrics
    query = f"""
    WITH latest_pb AS (
        SELECT 
            c.symbol,
            d.date,
            m.pb_ratio,
            m.close_price,
            ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY d.date DESC) as rn
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        WHERE c.symbol IN ({','.join([f"'{s}'" for s in symbols])})
          AND m.pb_ratio IS NOT NULL
    )
    SELECT symbol, date, pb_ratio, close_price
    FROM latest_pb
    WHERE rn <= 5  -- Get last 5 days for each symbol
    ORDER BY symbol, date DESC
    """
    
    pb_data = loader.query_to_dataframe(query)
    
    # Lowercase column names
    pb_data.columns = pb_data.columns.str.lower()
    
    print(f"\nLoaded {len(pb_data)} P/B ratio records")
    
    # Show the data we're using
    print("\nMost recent P/B ratios:")
    print("-" * 60)
    for symbol in symbols:
        symbol_data = pb_data[pb_data['symbol'] == symbol]
        if len(symbol_data) > 0:
            latest = symbol_data.iloc[0]
            print(f"{symbol}: P/B = {latest['pb_ratio']:.2f} (as of {latest['date']})")
    
    # Create and test factor
    factor = BookToMarket()
    
    print(f"\n\nCalculating {factor.name} factor...")
    print("Formula: B/M = 1 / P/B")
    
    # Calculate with diagnostics
    values, diagnostics = factor.calculate_with_diagnostics(pb_data)
    
    print("\n" + "-"*40)
    print("FACTOR VALUES")
    print("-"*40)
    print(f"{'Symbol':<8} {'P/B':<10} {'B/M':<10} {'Interpretation':<20}")
    print("-"*40)
    
    for symbol in symbols:
        if symbol in values.index:
            # Get the P/B ratio used
            symbol_pb = pb_data[pb_data['symbol'] == symbol].iloc[0]['pb_ratio']
            bm = values[symbol]
            
            if bm > 1:
                interpretation = "Deep Value (B > M)"
            elif bm > 0.5:
                interpretation = "Value"
            elif bm > 0.2:
                interpretation = "Neutral"
            else:
                interpretation = "Growth (B << M)"
                
            print(f"{symbol:<8} {symbol_pb:<10.2f} {bm:<10.4f} {interpretation:<20}")
        else:
            print(f"{symbol:<8} {'N/A':<10} {'NaN':<10} {'No data':<20}")
    
    print("\n" + "-"*40)
    print("DIAGNOSTICS")
    print("-"*40)
    for key, value in diagnostics.items():
        if key != 'percentiles':
            print(f"{key}: {value}")
    
    if 'percentiles' in diagnostics and diagnostics['percentiles']['25%'] is not None:
        print("percentiles:")
        for pct, val in diagnostics['percentiles'].items():
            print(f"  {pct}: {val:.4f}")
    
    # Manual verification
    print("\n" + "-"*40)
    print("MANUAL VERIFICATION")
    print("-"*40)
    
    # Pick one company for verification
    verify_symbol = 'AAPL'
    verify_data = pb_data[pb_data['symbol'] == verify_symbol]
    if len(verify_data) > 0:
        pb_ratio = verify_data.iloc[0]['pb_ratio']
        manual_bm = 1.0 / pb_ratio
        
        print(f"Symbol: {verify_symbol}")
        print(f"P/B Ratio: {pb_ratio:.4f}")
        print(f"B/M Ratio: 1 / {pb_ratio:.4f} = {manual_bm:.4f}")
        print(f"Factor calculation: {values.get(verify_symbol, 'NaN')}")
    
    # Historical comparison
    print("\n" + "-"*40)
    print("HISTORICAL P/B RANGES (from earlier check)")
    print("-"*40)
    
    historical_ranges = {
        'AAPL': (25.36, 69.07, 41.99),  # (min, max, avg)
        'AMZN': (6.07, 23.96, 11.20),
        'GOOGL': (4.28, 8.18, 6.34),
        'MSFT': (8.71, 16.96, 12.63),
        'NVDA': (11.70, 67.79, 33.15)
    }
    
    for symbol, (min_pb, max_pb, avg_pb) in historical_ranges.items():
        min_bm = 1/max_pb  # Min B/M when P/B is max
        max_bm = 1/min_pb  # Max B/M when P/B is min
        avg_bm = 1/avg_pb
        print(f"{symbol}: B/M range [{min_bm:.4f}, {max_bm:.4f}], avg: {avg_bm:.4f}")
    
    # Reasonableness checks
    print("\n" + "-"*40)
    print("REASONABLENESS CHECKS")
    print("-"*40)
    
    valid_values = values.dropna()
    if len(valid_values) > 0:
        print(f"✓ Calculated values for {len(valid_values)}/{len(symbols)} symbols")
        print(f"✓ All B/M values positive? {all(v > 0 for v in valid_values)}")
        print(f"✓ All B/M values < 1? {all(v < 1 for v in valid_values)}")
        print(f"✓ Any extreme values (>0.5)? {any(v > 0.5 for v in valid_values)}")
        
        # Given the high P/B ratios we saw, most B/M should be quite low
        very_low = sum(1 for v in valid_values if v < 0.1)
        print(f"✓ Very low B/M (<0.1): {very_low}/{len(valid_values)}")
    else:
        print("⚠️ No valid values calculated!")
    
    connector.disconnect()
    print("\n✓ Book-to-market factor test completed!")


if __name__ == "__main__":
    main()