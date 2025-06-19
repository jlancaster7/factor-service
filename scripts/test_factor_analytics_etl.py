#!/usr/bin/env python
"""
Test the Factor Analytics ETL Pipeline
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from loguru import logger

from src.config import Config
from src.etl.factor_analytics_etl import FactorAnalyticsETL


def verify_analytics_results(etl, test_date):
    """Verify analytics data was calculated correctly"""
    logger.info("\n=== VERIFYING ANALYTICS RESULTS ===")
    
    # Check analytics table
    query = """
    SELECT 
        f.factor_name,
        COUNT(*) as record_count,
        AVG(fa.z_score) as avg_z_score,
        STDDEV(fa.z_score) as std_z_score,
        MIN(fa.percentile_rank) as min_percentile,
        MAX(fa.percentile_rank) as max_percentile,
        AVG(fa.raw_value) as avg_raw_value
    FROM ANALYTICS.FACT_FACTOR_ANALYTICS fa
    JOIN ANALYTICS.DIM_DATE d ON fa.date_key = d.date_key
    JOIN ANALYTICS.DIM_FACTOR f ON fa.factor_key = f.factor_key
    WHERE d.date = %(date)s
    GROUP BY f.factor_name
    ORDER BY f.factor_name
    """
    
    results = etl.snowflake.fetch_all(query, {'date': test_date})
    
    logger.info("\nFACTOR ANALYTICS SUMMARY:")
    logger.info("-" * 80)
    for row in results:
        logger.info(f"{row['FACTOR_NAME']:20} | Records: {row['RECORD_COUNT']:3} | "
                   f"Avg Z-Score: {row['AVG_Z_SCORE']:7.4f} | "
                   f"Std Z-Score: {row['STD_Z_SCORE']:7.4f} | "
                   f"Percentile Range: [{row['MIN_PERCENTILE']:.2f}, {row['MAX_PERCENTILE']:.2f}]")
    
    # Verify z-scores properties (should be ~0 mean, ~1 std for each factor)
    logger.info("\n=== Z-SCORE VALIDATION ===")
    all_valid = True
    for row in results:
        factor_name = row['FACTOR_NAME']
        avg_z = row['AVG_Z_SCORE']
        std_z = row['STD_Z_SCORE']
        
        # Z-scores should have mean ~0 and std ~1
        mean_valid = abs(avg_z) < 0.01  # Allow small rounding error
        std_valid = abs(std_z - 1.0) < 0.1  # Allow some variance
        
        if mean_valid and std_valid:
            logger.success(f"✅ {factor_name}: Z-scores valid (mean={avg_z:.6f}, std={std_z:.4f})")
        else:
            logger.warning(f"⚠️  {factor_name}: Z-scores off (mean={avg_z:.6f}, std={std_z:.4f})")
            all_valid = False
    
    # Show sample records
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
    ORDER BY c.symbol, f.factor_name
    LIMIT 10
    """
    
    sample_results = etl.snowflake.fetch_all(sample_query, {'date': test_date})
    
    logger.info("\n=== SAMPLE ANALYTICS RECORDS ===")
    logger.info("Symbol | Factor              | Raw Value   | Z-Score  | Percentile")
    logger.info("-" * 70)
    for row in sample_results:
        logger.info(f"{row['SYMBOL']:6} | {row['FACTOR_NAME']:18} | "
                   f"{row['RAW_VALUE']:11.4f} | {row['Z_SCORE']:8.4f} | {row['PERCENTILE_RANK']:10.4f}")
    
    return all_valid


def compare_with_raw_data(etl, test_date):
    """Compare analytics records with raw data to ensure nothing was lost"""
    logger.info("\n=== COMPARING WITH RAW DATA ===")
    
    # Count raw records
    raw_query = """
    SELECT COUNT(*) as raw_count
    FROM ANALYTICS.FACT_FACTOR_RAW fr
    JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
    WHERE d.date = %(date)s
      AND fr.factor_value IS NOT NULL
    """
    
    raw_result = etl.snowflake.fetch_all(raw_query, {'date': test_date})
    raw_count = raw_result[0]['RAW_COUNT'] if raw_result else 0
    
    # Count analytics records
    analytics_query = """
    SELECT COUNT(*) as analytics_count
    FROM ANALYTICS.FACT_FACTOR_ANALYTICS fa
    JOIN ANALYTICS.DIM_DATE d ON fa.date_key = d.date_key
    WHERE d.date = %(date)s
    """
    
    analytics_result = etl.snowflake.fetch_all(analytics_query, {'date': test_date})
    analytics_count = analytics_result[0]['ANALYTICS_COUNT'] if analytics_result else 0
    
    logger.info(f"Raw records: {raw_count}")
    logger.info(f"Analytics records: {analytics_count}")
    
    if raw_count == analytics_count:
        logger.success("✅ All raw records have been processed into analytics")
    else:
        logger.warning(f"⚠️  Record count mismatch: {raw_count - analytics_count} records missing")
        
        # Find what's missing
        missing_query = """
        SELECT 
            c.symbol,
            f.factor_name,
            fr.factor_value
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
        JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
        WHERE d.date = %(date)s
          AND NOT EXISTS (
            SELECT 1 
            FROM ANALYTICS.FACT_FACTOR_ANALYTICS fa
            WHERE fa.company_key = fr.company_key
              AND fa.date_key = fr.date_key
              AND fa.factor_key = fr.factor_key
          )
        """
        
        missing = etl.snowflake.fetch_all(missing_query, {'date': test_date})
        if missing:
            logger.warning(f"Missing records: {missing[:5]}")  # Show first 5


def main():
    """Test the Factor Analytics ETL"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info("Starting Factor Analytics ETL Test")
    
    # Initialize config
    config = Config()
    config.validate()
    
    # Create ETL instance
    etl = FactorAnalyticsETL(config)
    
    # Use the same test date as our previous tests
    test_date = '2024-01-12'
    
    logger.info(f"Processing analytics for date: {test_date}")
    
    try:
        # First, check if we have raw data
        check_query = """
        SELECT 
            COUNT(DISTINCT c.symbol) as symbol_count,
            COUNT(DISTINCT f.factor_name) as factor_count,
            COUNT(*) as total_records
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
        JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
        WHERE d.date = %(date)s
        """
        
        raw_stats = etl.snowflake.fetch_all(check_query, {'date': test_date})
        if raw_stats:
            stats = raw_stats[0]
            logger.info(f"\nRaw data available: {stats['SYMBOL_COUNT']} symbols, "
                       f"{stats['FACTOR_COUNT']} factors, {stats['TOTAL_RECORDS']} total records")
        
        # Run the analytics ETL
        logger.info("\n=== RUNNING ANALYTICS ETL ===")
        result = etl.run(test_date)
        
        # Display results
        logger.info("\n=== ETL RESULTS ===")
        logger.info(f"Status: {result['status']}")
        logger.info(f"Records extracted: {result['records_extracted']}")
        logger.info(f"Records transformed: {result['records_transformed']}")
        logger.info(f"Records loaded: {result['records_loaded']}")
        
        if result['errors']:
            logger.warning(f"Errors: {result['errors']}")
        
        # Verify results
        if result['records_loaded'] > 0:
            z_scores_valid = verify_analytics_results(etl, test_date)
            compare_with_raw_data(etl, test_date)
            
            if result['status'] in ['SUCCESS', 'PARTIAL'] and z_scores_valid:
                logger.success("\n✅ Factor Analytics ETL test completed successfully!")
            else:
                logger.warning("\n⚠️  Factor Analytics ETL completed with warnings")
        else:
            logger.warning("\n⚠️  No records were loaded - check if raw data exists")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise
    finally:
        # Close connection
        if hasattr(etl, 'snowflake'):
            etl.snowflake.disconnect()


if __name__ == "__main__":
    main()