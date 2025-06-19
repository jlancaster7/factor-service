#!/usr/bin/env python
"""
Test the complete factor pipeline end-to-end:
1. FactorCalculationETL -> FACT_FACTOR_RAW
2. FactorAnalyticsETL -> FACT_FACTOR_ANALYTICS
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from loguru import logger

from src.config import Config
from src.etl.factor_calculation_etl import FactorCalculationETL
from src.etl.factor_analytics_etl import FactorAnalyticsETL


def clear_test_data(snowflake_conn, test_date, test_symbols):
    """Clear existing test data for a clean run"""
    logger.info("Clearing existing test data...")
    
    # Build placeholders and params
    placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(test_symbols))])
    params = {'date': test_date}
    for i, symbol in enumerate(test_symbols):
        params[f'symbol_{i}'] = symbol
    
    # Clear raw fact table
    delete_raw = """
    DELETE FROM ANALYTICS.FACT_FACTOR_RAW
    WHERE date_key = (SELECT date_key FROM ANALYTICS.DIM_DATE WHERE date = %(date)s)
      AND company_key IN (
        SELECT company_key FROM ANALYTICS.DIM_COMPANY 
        WHERE symbol IN ({placeholders})
      )
    """
    delete_raw = delete_raw.format(placeholders=placeholders)
    snowflake_conn.execute(delete_raw, params)
    
    # Clear analytics fact table
    delete_analytics = """
    DELETE FROM ANALYTICS.FACT_FACTOR_ANALYTICS
    WHERE date_key = (SELECT date_key FROM ANALYTICS.DIM_DATE WHERE date = %(date)s)
      AND company_key IN (
        SELECT company_key FROM ANALYTICS.DIM_COMPANY 
        WHERE symbol IN ({placeholders})
      )
    """
    delete_analytics = delete_analytics.format(placeholders=placeholders)
    snowflake_conn.execute(delete_analytics, params)
    
    logger.info("Test data cleared")


def verify_pipeline_results(snowflake_conn, test_date, test_symbols):
    """Verify data flowed through the entire pipeline"""
    logger.info("\n=== VERIFYING COMPLETE PIPELINE ===")
    
    # Check each stage (no more staging)
    stages = {
        'Raw Fact': 'ANALYTICS.FACT_FACTOR_RAW',
        'Analytics Fact': 'ANALYTICS.FACT_FACTOR_ANALYTICS'
    }
    
    results = {}
    
    for stage_name, table in stages.items():
        query = """
        SELECT COUNT(*) as count
        FROM {table} t
        JOIN ANALYTICS.DIM_DATE d ON t.date_key = d.date_key
        JOIN ANALYTICS.DIM_COMPANY c ON t.company_key = c.company_key
        WHERE d.date = %(date)s
          AND c.symbol IN ({placeholders})
        """
        
        placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(test_symbols))])
        query = query.format(table=table, placeholders=placeholders)
        
        params = {'date': test_date}
        for i, symbol in enumerate(test_symbols):
            params[f'symbol_{i}'] = symbol
        
        result = snowflake_conn.fetch_all(query, params)
        count = result[0]['COUNT'] if result else 0
        results[stage_name] = count
        
        logger.info(f"{stage_name:15} | Records: {count}")
    
    # Verify consistency
    if results['Raw Fact'] == results['Analytics Fact']:
        logger.success("✅ Data consistency verified across all stages!")
    else:
        logger.warning("⚠️  Data inconsistency detected between stages")
    
    # Show sample of final analytics data
    sample_query = """
    SELECT 
        c.symbol,
        f.factor_name,
        fa.raw_value,
        fa.z_score,
        fa.percentile_rank
    FROM ANALYTICS.FACT_FACTOR_ANALYTICS fa
    JOIN ANALYTICS.DIM_DATE d ON fa.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fa.factor_key = f.factor_key
    JOIN ANALYTICS.DIM_COMPANY c ON fa.company_key = c.company_key
    WHERE d.date = %(date)s
    ORDER BY f.factor_name, fa.z_score DESC
    """
    
    sample_results = snowflake_conn.fetch_all(sample_query, {'date': test_date})
    
    logger.info("\n=== FINAL ANALYTICS DATA ===")
    logger.info("Factor         | High Value Symbol | Z-Score | Low Value Symbol | Z-Score")
    logger.info("-" * 80)
    
    # Group by factor
    factors = {}
    for row in sample_results:
        factor = row['FACTOR_NAME']
        if factor not in factors:
            factors[factor] = []
        factors[factor].append(row)
    
    for factor, rows in factors.items():
        if len(rows) >= 2:
            high = rows[0]  # Highest z-score
            low = rows[-1]  # Lowest z-score
            logger.info(f"{factor:14} | {high['SYMBOL']:16} | {high['Z_SCORE']:7.4f} | "
                       f"{low['SYMBOL']:16} | {low['Z_SCORE']:7.4f}")


def main():
    """Test the complete factor pipeline"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info("Starting Complete Pipeline Test")
    
    # Initialize config
    config = Config()
    config.validate()
    
    # Test parameters
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    test_date = '2024-01-12'
    
    logger.info(f"Testing with {len(test_symbols)} symbols: {test_symbols}")
    logger.info(f"Calculation date: {test_date}")
    
    # Create ETL instances
    calc_etl = FactorCalculationETL(config)
    analytics_etl = FactorAnalyticsETL(config)
    
    try:
        # Clear existing test data
        clear_test_data(calc_etl.snowflake, test_date, test_symbols)
        
        # Step 1: Run Factor Calculation ETL
        logger.info("\n=== STEP 1: FACTOR CALCULATION ETL ===")
        calc_result = calc_etl.run(
            symbols=test_symbols,
            start_date=test_date,
            end_date=test_date
        )
        
        logger.info(f"Status: {calc_result['status']}")
        logger.info(f"Records extracted: {calc_result['records_extracted']}")
        logger.info(f"Records transformed: {calc_result['records_transformed']}")
        logger.info(f"Records loaded: {calc_result['records_loaded']}")
        
        if calc_result['errors']:
            logger.warning(f"Errors during calculation: {calc_result['errors']}")
        
        # Step 2: Run Analytics ETL
        logger.info("\n=== STEP 2: FACTOR ANALYTICS ETL ===")
        analytics_result = analytics_etl.run(test_date)
        
        logger.info(f"Status: {analytics_result['status']}")
        logger.info(f"Records extracted: {analytics_result['records_extracted']}")
        logger.info(f"Records transformed: {analytics_result['records_transformed']}")
        logger.info(f"Records loaded: {analytics_result['records_loaded']}")
        
        if analytics_result['errors']:
            logger.warning(f"Errors during analytics: {analytics_result['errors']}")
        
        # Verify results
        verify_pipeline_results(calc_etl.snowflake, test_date, test_symbols)
        
        # Check factor statistics
        if 'factor_stats' in calc_result:
            logger.info("\n=== FACTOR CALCULATION SUMMARY ===")
            for factor, stats in calc_result['factor_stats'].items():
                if 'error' not in stats:
                    logger.info(f"{factor}: {stats['calculated']} values calculated, "
                               f"{stats['nulls']} nulls, {stats['infs']} infs")
                else:
                    logger.warning(f"{factor}: ERROR - {stats['error']}")
        
        # Overall success check
        if (calc_result['status'] in ['SUCCESS', 'PARTIAL'] and 
            analytics_result['status'] in ['SUCCESS', 'PARTIAL']):
            logger.success("\n✅ Complete pipeline test PASSED!")
        else:
            logger.error("\n❌ Complete pipeline test FAILED!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise
    finally:
        # Close connections
        if hasattr(calc_etl, 'snowflake'):
            calc_etl.snowflake.disconnect()
        if hasattr(analytics_etl, 'snowflake'):
            analytics_etl.snowflake.disconnect()


if __name__ == "__main__":
    main()