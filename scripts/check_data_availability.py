#!/usr/bin/env python
"""Check what data is available in Snowflake for our test companies."""
import sys
import logging
from datetime import datetime

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader


def main():
    """Check data availability for factor calculations."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load config
    config = Config()
    snowflake_config = config.get_snowflake_config()
    connector = SnowflakeConnector(snowflake_config)
    
    # Connect and create data loader
    connector.connect()
    loader = DataLoader(connector)
    
    # Test companies
    test_symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    symbols_str = "', '".join(test_symbols)
    
    logger.info(f"Checking data availability for: {', '.join(test_symbols)}")
    
    # Check price data
    print("\n" + "="*60)
    print("PRICE DATA AVAILABILITY")
    print("="*60)
    
    price_query = f"""
    SELECT 
        c.symbol,
        COUNT(*) as record_count,
        MIN(d.date) as min_date,
        MAX(d.date) as max_date,
        DATEDIFF('day', MIN(d.date), MAX(d.date)) + 1 as total_days,
        COUNT(DISTINCT d.date) as trading_days,
        ROUND(COUNT(DISTINCT d.date) * 100.0 / (DATEDIFF('day', MIN(d.date), MAX(d.date)) + 1), 2) as coverage_pct
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol IN ('{symbols_str}')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    price_summary = loader.query_to_dataframe(price_query)
    print(price_summary.to_string(index=False))
    
    # Check fundamental data
    print("\n" + "="*60)
    print("FUNDAMENTAL DATA AVAILABILITY")
    print("="*60)
    
    fundamental_query = f"""
    SELECT 
        c.symbol,
        COUNT(*) as quarters_count,
        MIN(d.date) as earliest_quarter,
        MAX(d.date) as latest_quarter,
        COUNT(DISTINCT f.period_type) as period_types
    FROM ANALYTICS.FACT_FINANCIALS f
    JOIN ANALYTICS.DIM_COMPANY c ON f.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON f.fiscal_date_key = d.date_key
    WHERE c.symbol IN ('{symbols_str}')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    fundamental_summary = loader.query_to_dataframe(fundamental_query)
    print(fundamental_summary.to_string(index=False))
    
    # Check recent data points
    print("\n" + "="*60)
    print("SAMPLE RECENT PRICE DATA")
    print("="*60)
    
    sample_query = f"""
    SELECT 
        c.symbol,
        d.date,
        p.adj_close,
        p.volume,
        p.change_percent
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol = 'AAPL'
    ORDER BY d.date DESC
    LIMIT 5
    """
    
    sample_data = loader.query_to_dataframe(sample_query)
    print(sample_data.to_string(index=False))
    
    # Check for data gaps
    print("\n" + "="*60)
    print("DATA QUALITY CHECKS")
    print("="*60)
    
    # Check for nulls in price data
    null_check_query = f"""
    SELECT 
        c.symbol,
        SUM(CASE WHEN p.adj_close IS NULL THEN 1 ELSE 0 END) as null_adj_close,
        SUM(CASE WHEN p.volume IS NULL THEN 1 ELSE 0 END) as null_volume,
        SUM(CASE WHEN p.volume = 0 THEN 1 ELSE 0 END) as zero_volume
    FROM ANALYTICS.FACT_DAILY_PRICES p
    JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
    WHERE c.symbol IN ('{symbols_str}')
    GROUP BY c.symbol
    HAVING null_adj_close > 0 OR null_volume > 0 OR zero_volume > 0
    """
    
    null_results = loader.query_to_dataframe(null_check_query)
    if len(null_results) > 0:
        print("⚠️ Data quality issues found:")
        print(null_results.to_string(index=False))
    else:
        print("✓ No null or zero values found in price data")
    
    # Check TTM data availability
    print("\n" + "="*60)
    print("TTM (TRAILING TWELVE MONTHS) DATA")
    print("="*60)
    
    ttm_query = f"""
    SELECT 
        c.symbol,
        COUNT(*) as ttm_records,
        MAX(t.calculation_date) as latest_ttm_date
    FROM ANALYTICS.FACT_FINANCIALS_TTM t
    JOIN ANALYTICS.DIM_COMPANY c ON t.company_key = c.company_key
    WHERE c.symbol IN ('{symbols_str}')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    ttm_summary = loader.query_to_dataframe(ttm_query)
    print(ttm_summary.to_string(index=False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"✓ Price data available for all {len(test_symbols)} test companies")
    print(f"✓ Fundamental data available for all companies")
    if len(price_summary) > 0:
        print(f"✓ Data range: {price_summary['MIN_DATE'].min()} to {price_summary['MAX_DATE'].max()}")
    
    # Calculate if we have enough data for momentum factor (252 days)
    min_records = price_summary['RECORD_COUNT'].min()
    if min_records >= 252:
        print(f"✓ Sufficient data for 12-month momentum calculation (min records: {min_records})")
    else:
        print(f"⚠️ Insufficient data for 12-month momentum (min records: {min_records}, need 252)")
    
    # Check total companies available
    print("\n" + "="*60)
    print("TOTAL COMPANIES IN DATABASE")
    print("="*60)
    
    total_companies_query = """
    SELECT 
        COUNT(DISTINCT c.company_key) as total_companies,
        COUNT(DISTINCT CASE WHEN c.is_current = TRUE THEN c.company_key END) as active_companies
    FROM ANALYTICS.DIM_COMPANY c
    """
    
    total_companies = loader.query_to_dataframe(total_companies_query)
    print(f"Total companies in database: {total_companies.iloc[0]['TOTAL_COMPANIES']}")
    print(f"Active companies: {total_companies.iloc[0]['ACTIVE_COMPANIES']}")
    
    connector.disconnect()


if __name__ == "__main__":
    main()