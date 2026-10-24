#!/usr/bin/env python3
"""
Validate factor data quality and coverage.
"""
import sys
sys.path.append('.')

from src.config import Config
from src.data.database import DatabaseConnector
import pandas as pd

def main():
    config = Config()
    conn = DatabaseConnector(config)
    conn.connect()
    
    print("=" * 80)
    print("FACTOR DATA VALIDATION REPORT")
    print("=" * 80)
    
    # 1. Overall coverage
    query = """
    SELECT 
        COUNT(DISTINCT c.symbol) as symbols,
        COUNT(DISTINCT f.factor_name) as factors,
        COUNT(DISTINCT d.date) as dates,
        MIN(d.date) as start_date,
        MAX(d.date) as end_date,
        COUNT(*) as total_records
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
    """
    
    coverage = conn.fetch_all(query)[0]
    print("\n📊 Overall Coverage:")
    print(f"  Symbols: {coverage['SYMBOLS']}")
    print(f"  Factors: {coverage['FACTORS']}")
    print(f"  Trading days: {coverage['DATES']:,}")
    print(f"  Date range: {coverage['START_DATE']} to {coverage['END_DATE']}")
    print(f"  Total records: {coverage['TOTAL_RECORDS']:,}")
    
    # 2. Factor-level statistics
    query = """
    SELECT 
        f.factor_name,
        COUNT(*) as record_count,
        COUNT(DISTINCT c.symbol) as symbols_covered,
        COUNT(DISTINCT d.date) as dates_covered,
        MIN(d.date) as first_date,
        MAX(d.date) as last_date,
        AVG(fr.factor_value) as avg_value,
        STDDEV(fr.factor_value) as std_value,
        MIN(fr.factor_value) as min_value,
        MAX(fr.factor_value) as max_value
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
    GROUP BY f.factor_name
    ORDER BY f.factor_name
    """
    
    factor_stats = conn.fetch_all(query)
    print("\n📈 Factor Statistics:")
    for row in factor_stats:
        print(f"\n  {row['FACTOR_NAME']}:")
        print(f"    Records: {row['RECORD_COUNT']:,}")
        print(f"    Symbols: {row['SYMBOLS_COVERED']}")
        print(f"    Dates: {row['DATES_COVERED']:,}")
        print(f"    Range: [{row['MIN_VALUE']:.4f}, {row['MAX_VALUE']:.4f}]")
        print(f"    Mean: {row['AVG_VALUE']:.4f} (Std: {row['STD_VALUE']:.4f})")
        
        # Factor-specific validation
        if row['FACTOR_NAME'] == 'rsi_14':
            if row['MIN_VALUE'] < 0 or row['MAX_VALUE'] > 100:
                print(f"    ⚠️  WARNING: RSI values outside [0, 100] range!")
        elif row['FACTOR_NAME'] == 'book_to_market':
            if row['MIN_VALUE'] < 0:
                print(f"    ⚠️  Note: {row['MIN_VALUE']:.2f} negative B/M (distressed companies)")
    
    # 3. Coverage by symbol
    query = """
    SELECT 
        c.symbol,
        COUNT(DISTINCT f.factor_name) as factors_available,
        COUNT(DISTINCT d.date) as dates_available,
        MIN(d.date) as first_date,
        MAX(d.date) as last_date,
        COUNT(*) as total_records
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
    GROUP BY c.symbol
    ORDER BY total_records DESC
    LIMIT 10
    """
    
    symbol_coverage = conn.fetch_all(query)
    print("\n📊 Top 10 Symbols by Coverage:")
    for row in symbol_coverage:
        print(f"  {row['SYMBOL']:6} - {row['TOTAL_RECORDS']:,} records, "
              f"{row['FACTORS_AVAILABLE']} factors, {row['DATES_AVAILABLE']:,} days")
    
    # 4. Recent data check
    query = """
    SELECT 
        d.date,
        COUNT(DISTINCT c.symbol) as symbols_calculated,
        COUNT(DISTINCT f.factor_name) as factors_calculated,
        COUNT(*) as total_records
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
    WHERE d.date >= DATEADD(day, -10, CURRENT_DATE())
    GROUP BY d.date
    ORDER BY d.date DESC
    """
    
    recent_data = conn.fetch_all(query)
    print("\n📅 Recent Data (Last 10 Days):")
    for row in recent_data:
        print(f"  {row['DATE']}: {row['SYMBOLS_CALCULATED']} symbols, "
              f"{row['FACTORS_CALCULATED']} factors, {row['TOTAL_RECORDS']} values")
    
    # 5. Data quality issues
    query = """
    WITH factor_issues AS (
        SELECT 
            f.factor_name,
            SUM(CASE WHEN fr.factor_value < 0 AND f.factor_name = 'rsi_14' THEN 1 ELSE 0 END) as rsi_negative,
            SUM(CASE WHEN fr.factor_value > 100 AND f.factor_name = 'rsi_14' THEN 1 ELSE 0 END) as rsi_over_100,
            SUM(CASE WHEN ABS(fr.factor_value) > 10 AND f.factor_name = 'momentum_12_1' THEN 1 ELSE 0 END) as extreme_momentum,
            COUNT(*) as total_records
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
        GROUP BY f.factor_name
    )
    SELECT * FROM factor_issues
    WHERE rsi_negative > 0 OR rsi_over_100 > 0 OR extreme_momentum > 0
    """
    
    issues = conn.fetch_all(query)
    if issues:
        print("\n⚠️  Data Quality Issues:")
        for row in issues:
            if row['RSI_NEGATIVE'] > 0:
                print(f"  {row['FACTOR_NAME']}: {row['RSI_NEGATIVE']} negative values")
            if row['RSI_OVER_100'] > 0:
                print(f"  {row['FACTOR_NAME']}: {row['RSI_OVER_100']} values > 100")
            if row['EXTREME_MOMENTUM'] > 0:
                print(f"  {row['FACTOR_NAME']}: {row['EXTREME_MOMENTUM']} extreme values (>1000% change)")
    else:
        print("\n✅ No data quality issues detected!")
    
    conn.disconnect()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()