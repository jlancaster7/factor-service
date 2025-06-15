#!/usr/bin/env python
"""Calculate all registered factors for test companies."""
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader
from src.factors.registry import FactorRegistry

# Import all factor modules to register them
import src.factors.momentum
import src.factors.value
import src.factors.technical


def validate_results(results_df: pd.DataFrame):
    """Validate factor calculation results."""
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    for col in results_df.columns:
        if col == 'symbol':
            continue
            
        print(f"\n{col}:")
        valid_data = results_df[col].dropna()
        
        print(f"  Non-null: {len(valid_data)}/{len(results_df)} ({len(valid_data)/len(results_df)*100:.1f}%)")
        
        if len(valid_data) > 0:
            print(f"  Range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
            print(f"  Mean: {valid_data.mean():.4f}")
            print(f"  Std: {valid_data.std():.4f}")
            
            # Factor-specific checks
            if col == 'rsi_14':
                out_of_range = ((valid_data < 0) | (valid_data > 100)).sum()
                if out_of_range > 0:
                    print(f"  ⚠️  WARNING: {out_of_range} values outside [0, 100] range!")
            elif col == 'book_to_market':
                negative = (valid_data < 0).sum()
                if negative > 0:
                    print(f"  ⚠️  WARNING: {negative} negative B/M values (distressed companies)")
            elif col == 'momentum_12_1':
                extreme = (valid_data.abs() > 2).sum()
                if extreme > 0:
                    print(f"  ⚠️  WARNING: {extreme} extreme momentum values (>200% change)")
        else:
            print("  ⚠️  WARNING: All values are null!")


def main():
    """Calculate all factors for test companies."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    config = Config()
    config.validate()
    
    # Test companies (could expand to all 50 later)
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    
    # Date range
    end_date = datetime.now().date()
    momentum_start = end_date - timedelta(days=365 * 2)  # 2 years for momentum
    recent_start = end_date - timedelta(days=60)  # 60 days for RSI
    
    print("\n" + "="*60)
    print("EQUITY FACTOR CALCULATION SERVICE")
    print("="*60)
    print(f"\nCalculation Date: {end_date}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Registered Factors: {', '.join(FactorRegistry.list_factors())}")
    
    # Connect to Snowflake
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    connector.connect()
    loader = DataLoader(connector)
    
    # Load data for different factors
    print("\n" + "-"*40)
    print("LOADING DATA")
    print("-"*40)
    
    # 1. Price data for momentum and RSI
    print(f"Loading price data from {momentum_start} to {end_date}...")
    price_query = f"""
    SELECT 
        c.symbol,
        d.date,
        p.adj_close
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol IN ({','.join([f"'{s}'" for s in symbols])})
      AND d.date BETWEEN '{momentum_start}' AND '{end_date}'
    ORDER BY c.symbol, d.date
    """
    price_data = loader.query_to_dataframe(price_query)
    price_data.columns = price_data.columns.str.lower()
    print(f"Loaded {len(price_data)} price records")
    
    # 2. Market metrics for book-to-market
    print(f"\nLoading market metrics...")
    pb_query = f"""
    WITH latest_pb AS (
        SELECT 
            c.symbol,
            d.date,
            m.pb_ratio,
            ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY d.date DESC) as rn
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        WHERE c.symbol IN ({','.join([f"'{s}'" for s in symbols])})
          AND m.pb_ratio IS NOT NULL
    )
    SELECT symbol, date, pb_ratio
    FROM latest_pb
    WHERE rn = 1
    """
    pb_data = loader.query_to_dataframe(pb_query)
    pb_data.columns = pb_data.columns.str.lower()
    print(f"Loaded P/B ratios for {len(pb_data)} companies")
    
    # Calculate factors
    print("\n" + "-"*40)
    print("CALCULATING FACTORS")
    print("-"*40)
    
    results = pd.DataFrame({'symbol': symbols})
    results.set_index('symbol', inplace=True)
    
    all_diagnostics = {}
    
    # Calculate each registered factor
    for factor_name in FactorRegistry.list_factors():
        print(f"\nCalculating {factor_name}...")
        
        try:
            factor = FactorRegistry.create(factor_name)
            
            # Determine which data to use
            if factor.category in ['momentum', 'technical']:
                data = price_data
            else:  # value factors
                data = pb_data
            
            # Calculate with diagnostics
            values, diagnostics = factor.calculate_with_diagnostics(data)
            
            # Add to results
            results[factor_name] = values
            all_diagnostics[factor_name] = diagnostics
            
            # Show quick summary
            valid_count = values.notna().sum()
            print(f"  ✓ Calculated for {valid_count}/{len(symbols)} symbols")
            if valid_count > 0:
                print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
            
        except Exception as e:
            logger.error(f"Failed to calculate {factor_name}: {e}")
            results[factor_name] = None
    
    # Display results
    print("\n" + "="*60)
    print("FACTOR VALUES SUMMARY")
    print("="*60)
    
    # Format for nice display
    display_df = results.copy()
    
    # Format each column appropriately
    for col in display_df.columns:
        if col == 'momentum_12_1':
            # Show as percentage
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        elif col == 'book_to_market':
            # Show with 4 decimals
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        elif col == 'rsi_14':
            # Show with 2 decimals
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    print(display_df)
    
    # Validate results
    validate_results(results)
    
    # Save results
    output_file = f"factor_values_{end_date}.csv"
    results.to_csv(output_file)
    print(f"\n✓ Results saved to {output_file}")
    
    # Show cross-factor analysis
    print("\n" + "="*60)
    print("CROSS-FACTOR ANALYSIS")
    print("="*60)
    
    # Convert back to numeric for analysis
    analysis_df = results.copy()
    
    # Find interesting combinations
    print("\nInteresting observations:")
    
    for symbol in symbols:
        observations = []
        
        # Get values
        momentum = analysis_df.loc[symbol, 'momentum_12_1']
        bm = analysis_df.loc[symbol, 'book_to_market']
        rsi = analysis_df.loc[symbol, 'rsi_14']
        
        # Check patterns
        if pd.notna(momentum) and pd.notna(rsi):
            if momentum > 0.2 and rsi > 70:
                observations.append("Strong momentum + Overbought")
            elif momentum < -0.1 and rsi < 30:
                observations.append("Negative momentum + Oversold")
        
        if pd.notna(bm):
            if bm < 0.05:
                observations.append("Very low B/M (growth)")
            elif bm > 0.5:
                observations.append("High B/M (value)")
        
        if observations:
            print(f"\n{symbol}: {', '.join(observations)}")
    
    # Correlation analysis (if enough data)
    numeric_results = results.select_dtypes(include=[float, int])
    if len(numeric_results.dropna()) >= 3:
        print("\n" + "-"*40)
        print("FACTOR CORRELATIONS")
        print("-"*40)
        corr_matrix = numeric_results.corr()
        print(corr_matrix.round(3))
    
    connector.disconnect()
    print("\n✓ Factor calculation completed!")


if __name__ == "__main__":
    main()