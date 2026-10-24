#!/usr/bin/env python3
"""
Check what data is available in the database for factor calculations.
"""
import sys
sys.path.append('.')

from src.config import Config
from src.data.database import DatabaseConnector
from src.data.data_loader import DataLoader

def main():
    config = Config()
    snowflake = DatabaseConnector(config)
    snowflake.connect()
    data_loader = DataLoader(snowflake)
    
    try:
        print("=" * 80)
        print("CHECKING AVAILABLE DATA FOR FACTOR CALCULATIONS")
        print("=" * 80)
        
        # Check company count
        query = """
        SELECT COUNT(DISTINCT company_key) as company_count
        FROM ANALYTICS.DIM_COMPANY
        WHERE is_current = TRUE
        """
        result = data_loader.query_to_dataframe(query)
        print(f"\nActive companies in DIM_COMPANY: {result['company_count'].iloc[0]}")
        
        # Check price data coverage
        query = """
        SELECT 
            COUNT(DISTINCT c.symbol) as symbol_count,
            MIN(d.date) as start_date,
            MAX(d.date) as end_date,
            COUNT(DISTINCT d.date) as trading_days,
            COUNT(*) as total_records
        FROM ANALYTICS.FACT_DAILY_PRICES p
        JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
        """
        price_coverage = data_loader.query_to_dataframe(query)
        print(f"\n📊 FACT_DAILY_PRICES Coverage:")
        print(f"  Symbols: {price_coverage['symbol_count'].iloc[0]}")
        print(f"  Date range: {price_coverage['start_date'].iloc[0]} to {price_coverage['end_date'].iloc[0]}")
        print(f"  Trading days: {price_coverage['trading_days'].iloc[0]}")
        print(f"  Total records: {price_coverage['total_records'].iloc[0]:,}")
        
        # Check market metrics coverage
        query = """
        SELECT 
            COUNT(DISTINCT c.symbol) as symbol_count,
            MIN(d.date) as start_date,
            MAX(d.date) as end_date,
            COUNT(DISTINCT d.date) as trading_days,
            COUNT(*) as total_records
        FROM ANALYTICS.FACT_MARKET_METRICS m
        JOIN ANALYTICS.DIM_COMPANY c ON m.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON m.date_key = d.date_key
        """
        market_coverage = data_loader.query_to_dataframe(query)
        print(f"\n📈 FACT_MARKET_METRICS Coverage:")
        print(f"  Symbols: {market_coverage['symbol_count'].iloc[0]}")
        print(f"  Date range: {market_coverage['start_date'].iloc[0]} to {market_coverage['end_date'].iloc[0]}")
        print(f"  Trading days: {market_coverage['trading_days'].iloc[0]}")
        print(f"  Total records: {market_coverage['total_records'].iloc[0]:,}")
        
        # Check what's already been calculated
        query = """
        SELECT 
            COUNT(DISTINCT c.symbol) as symbol_count,
            COUNT(DISTINCT f.factor_name) as factor_count,
            MIN(d.date) as start_date,
            MAX(d.date) as end_date,
            COUNT(DISTINCT d.date) as dates_calculated,
            COUNT(*) as total_records
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
        """
        factor_coverage = data_loader.query_to_dataframe(query)
        print(f"\n🔢 FACT_FACTOR_RAW Coverage (Already Calculated):")
        if factor_coverage['total_records'].iloc[0] > 0:
            print(f"  Symbols: {factor_coverage['symbol_count'].iloc[0]}")
            print(f"  Factors: {factor_coverage['factor_count'].iloc[0]}")
            print(f"  Date range: {factor_coverage['start_date'].iloc[0]} to {factor_coverage['end_date'].iloc[0]}")
            print(f"  Dates calculated: {factor_coverage['dates_calculated'].iloc[0]}")
            print(f"  Total records: {factor_coverage['total_records'].iloc[0]:,}")
        else:
            print("  No factors calculated yet")
        
        # Get list of all symbols
        query = """
        SELECT DISTINCT c.symbol
        FROM ANALYTICS.DIM_COMPANY c
        WHERE c.is_current = TRUE
        ORDER BY c.symbol
        """
        symbols = data_loader.query_to_dataframe(query)
        print(f"\n📝 Available symbols ({len(symbols)}):")
        symbol_list = symbols['symbol'].tolist()
        # Print in columns
        for i in range(0, len(symbol_list), 10):
            print("  " + ", ".join(symbol_list[i:i+10]))
        
        # Estimate calculation scope
        print(f"\n🎯 CALCULATION SCOPE:")
        days_to_calculate = price_coverage['trading_days'].iloc[0]
        symbols_count = price_coverage['symbol_count'].iloc[0]
        factors_count = 3  # momentum_12_1, book_to_market, rsi_14
        total_calculations = days_to_calculate * symbols_count * factors_count
        
        print(f"  Trading days: {days_to_calculate:,}")
        print(f"  Symbols: {symbols_count}")
        print(f"  Factors: {factors_count}")
        print(f"  Total calculations needed: {total_calculations:,}")
        
        # Check if we need to backfill
        if factor_coverage['total_records'].iloc[0] > 0:
            already_calculated = factor_coverage['total_records'].iloc[0]
            remaining = total_calculations - already_calculated
            print(f"  Already calculated: {already_calculated:,}")
            print(f"  Remaining: {remaining:,}")
            print(f"  Progress: {(already_calculated/total_calculations)*100:.1f}%")
        
    finally:
        snowflake.disconnect()
        print("\n✅ Data check complete")

if __name__ == "__main__":
    main()