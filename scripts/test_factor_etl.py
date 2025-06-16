#!/usr/bin/env python
"""
Test script for FactorCalculationETL
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime, timedelta
from loguru import logger

from src.config import Config
from src.etl.factor_calculation_etl import FactorCalculationETL


def main():
    """Test the factor calculation ETL pipeline"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", 
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}")
    
    logger.info("Starting Factor Calculation ETL test")
    
    # Initialize config
    config = Config()
    config.validate()
    
    # Create ETL instance
    etl = FactorCalculationETL(config)
    
    # Test with our 5 companies for recent date
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    test_date = '2024-01-12'  # Use a valid trading day (Friday)
    
    logger.info(f"Testing with symbols: {test_symbols}")
    logger.info(f"Calculation date: {test_date}")
    
    try:
        # Run the ETL
        result = etl.run(
            symbols=test_symbols,
            start_date=test_date,
            end_date=test_date
        )
        
        # Display results
        logger.info("=" * 60)
        logger.info("ETL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Status: {result['status']}")
        logger.info(f"Records extracted: {result['records_extracted']}")
        logger.info(f"Records transformed: {result['records_transformed']}")
        logger.info(f"Records loaded: {result['records_loaded']}")
        
        if result['errors']:
            logger.warning(f"Errors encountered: {len(result['errors'])}")
            for error in result['errors']:
                logger.warning(f"  - {error}")
        
        # Display factor statistics
        logger.info("\nFACTOR STATISTICS:")
        factor_stats = result.get('factor_stats', {})
        for factor_name, stats in factor_stats.items():
            if 'error' in stats:
                logger.error(f"{factor_name}: ERROR - {stats['error']}")
            else:
                logger.info(f"{factor_name}:")
                logger.info(f"  - Calculated: {stats['calculated']}")
                logger.info(f"  - Nulls: {stats['nulls']}")
                logger.info(f"  - Infs: {stats['infs']}")
                if stats['mean'] is not None:
                    logger.info(f"  - Mean: {stats['mean']:.4f}")
                else:
                    logger.info(f"  - Mean: None")
                if stats['std'] is not None:
                    logger.info(f"  - Std: {stats['std']:.4f}")
                else:
                    logger.info(f"  - Std: None")
        
        # Verify data was loaded to staging
        logger.info("\nVerifying staging data...")
        verify_query = """
        SELECT 
            factor_name,
            COUNT(*) as record_count,
            AVG(factor_value) as avg_value,
            MIN(factor_value) as min_value,
            MAX(factor_value) as max_value
        FROM STAGING.STG_FACTOR_VALUES
        WHERE date = %(date)s
          AND symbol IN ({placeholders})
        GROUP BY factor_name
        ORDER BY factor_name
        """
        
        placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(test_symbols))])
        verify_query = verify_query.format(placeholders=placeholders)
        
        params = {'date': test_date}
        for i, symbol in enumerate(test_symbols):
            params[f'symbol_{i}'] = symbol
        
        verification = etl.snowflake.fetch_all(verify_query, params)
        
        logger.info("\nSTAGING TABLE VERIFICATION:")
        for row in verification:
            logger.info(f"{row['FACTOR_NAME']}:")
            logger.info(f"  - Records: {row['RECORD_COUNT']}")
            logger.info(f"  - Avg: {row['AVG_VALUE']:.4f}")
            logger.info(f"  - Min: {row['MIN_VALUE']:.4f}")
            logger.info(f"  - Max: {row['MAX_VALUE']:.4f}")
        
        if result['status'] == 'SUCCESS':
            logger.success("✅ Factor Calculation ETL test completed successfully!")
        else:
            logger.warning(f"⚠️  ETL completed with status: {result['status']}")
            
    except Exception as e:
        logger.error(f"❌ ETL test failed: {str(e)}")
        raise
    finally:
        # Close connection
        if hasattr(etl, 'snowflake'):
            etl.snowflake.disconnect()


if __name__ == "__main__":
    main()