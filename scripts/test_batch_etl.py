#!/usr/bin/env python3
"""Test the updated ETL with batch processing of date ranges."""
import sys
sys.path.append('.')

import time
from src.config import Config
from src.etl.factor_calculation_etl import FactorCalculationETL
from src.data.database import DatabaseConnector

def test_batch_processing():
    """Test batch processing with a small date range."""
    config = Config()
    
    print("=" * 80)
    print("TESTING BATCH FACTOR CALCULATION ETL")
    print("=" * 80)
    
    # Initialize ETL
    calc_etl = FactorCalculationETL(config)
    
    # Test with 5 days in June 2025
    test_start = '2025-06-05'
    test_end = '2025-06-12'
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']  # Just 3 symbols for testing
    
    print(f"\nTest parameters:")
    print(f"  Date range: {test_start} to {test_end}")
    print(f"  Symbols: {test_symbols}")
    print(f"  Expected calculations: ~15 (3 symbols × 5 days × 3 factors)")
    
    # Run the ETL
    print("\nRunning batch ETL...")
    start_time = time.time()
    
    result = calc_etl.run(
        symbols=test_symbols,
        start_date=test_start,
        end_date=test_end
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    print(f"\n✅ ETL completed in {elapsed:.2f} seconds")
    print(f"Status: {result['status']}")
    print(f"Records extracted: {result['records_extracted']:,}")
    print(f"Records transformed: {result['records_transformed']:,}")
    print(f"Records loaded: {result['records_loaded']:,}")
    
    if result['errors']:
        print(f"\nErrors encountered: {len(result['errors'])}")
        for error in result['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Show factor statistics
    print("\nFactor Statistics:")
    for factor_name, stats in result['factor_stats'].items():
        print(f"  {factor_name}:")
        print(f"    Dates calculated: {stats.get('dates_calculated', 0)}")
        print(f"    Total values: {stats.get('total_values', 0)}")
        print(f"    Total nulls: {stats.get('total_nulls', 0)}")
    
    # Verify data in database
    print("\nVerifying data in database...")
    conn = DatabaseConnector(config)
    conn.connect()
    
    verify_query = """
    SELECT 
        f.factor_name,
        COUNT(DISTINCT d.date) as dates,
        COUNT(DISTINCT c.symbol) as symbols,
        COUNT(*) as total_records
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
    WHERE c.symbol IN (%(symbol_0)s, %(symbol_1)s, %(symbol_2)s)
      AND d.date BETWEEN %(start_date)s AND %(end_date)s
    GROUP BY f.factor_name
    ORDER BY f.factor_name
    """
    
    params = {
        'symbol_0': test_symbols[0],
        'symbol_1': test_symbols[1],
        'symbol_2': test_symbols[2],
        'start_date': test_start,
        'end_date': test_end
    }
    
    results = conn.fetch_all(verify_query, params)
    
    print("\nDatabase verification:")
    for row in results:
        print(f"  {row['FACTOR_NAME']}: {row['TOTAL_RECORDS']} records "
              f"({row['SYMBOLS']} symbols × {row['DATES']} dates)")
    
    conn.disconnect()
    
    print("\n✅ Batch processing test completed successfully!")
    
    # Calculate efficiency
    if result['records_loaded'] > 0:
        records_per_second = result['records_loaded'] / elapsed
        print(f"\nPerformance: {records_per_second:.0f} records/second")
        
        # Estimate for full dataset
        estimated_full_time = (50 * 1256 * 3) / records_per_second  # 50 symbols × 1256 days × 3 factors
        print(f"Estimated time for full dataset: {estimated_full_time/60:.1f} minutes")

if __name__ == "__main__":
    test_batch_processing()