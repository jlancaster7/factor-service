"""
Factor Analytics ETL Pipeline
Processes raw factor values from FACT_FACTOR_RAW and calculates
cross-sectional statistics for FACT_FACTOR_ANALYTICS
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from loguru import logger

from src.etl.base_etl import BaseETL
from src.data.database import DatabaseConnector
from src.config import Config


class FactorAnalyticsETL(BaseETL):
    """ETL pipeline for calculating factor analytics (z-scores, percentiles)"""
    
    def __init__(self, config: Config):
        """
        Initialize Factor Analytics ETL
        
        Args:
            config: Application configuration object
        """
        database_connector = DatabaseConnector(config)
        database_connector.initialize_schema()
        
        super().__init__(
            job_name="factor_analytics_etl",
            database_connector=database_connector,
            fmp_client=None,  # Not needed
            batch_size=config.batch_size or 1000,
            max_retries=3,
            retry_delay=5,
            enable_monitoring=False
        )
        
        self.config = config
        self.processing_date = None
        
    def extract(self, calculation_date: str) -> List[Dict[str, Any]]:
        """
        Extract raw factor values from FACT_FACTOR_RAW for a specific date
        
        Args:
            calculation_date: Date to process (YYYY-MM-DD format)
            
        Returns:
            List of raw factor records
        """
        self.processing_date = calculation_date
        
        # Check if we have data to process
        check_query = """
        SELECT COUNT(*) as record_count
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        WHERE d.date = %(date)s
        """
        
        result = self.database.fetch_all(check_query, {'date': calculation_date})
        record_count = result[0]['RECORD_COUNT'] if result else 0
        
        if record_count == 0:
            logger.warning(f"No raw factor data found for {calculation_date}")
            return []
        
        logger.info(f"Found {record_count} raw factor records for {calculation_date}")
        
        # Extract raw factor values with necessary joins
        query = """
        SELECT 
            fr.company_key,
            fr.date_key,
            fr.factor_key,
            fr.factor_value as raw_value,
            c.symbol,
            c.sector,
            d.date,
            f.factor_name
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        JOIN ANALYTICS.DIM_FACTOR f ON fr.factor_key = f.factor_key
        WHERE d.date = %(date)s
          AND fr.factor_value IS NOT NULL
          AND c.is_current = TRUE
        ORDER BY f.factor_name, c.symbol
        """
        
        raw_data = self.database.fetch_all(query, {'date': calculation_date})
        
        # Store metadata
        self.result.metadata['calculation_date'] = calculation_date
        self.result.metadata['raw_records'] = len(raw_data)
        
        logger.info(f"Extracted {len(raw_data)} valid factor values")
        
        return raw_data
    
    def transform(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate cross-sectional statistics using SQL window functions
        
        Args:
            raw_data: List of raw factor records
            
        Returns:
            List of records with calculated statistics
        """
        if not raw_data:
            return []
        
        # Use SQL to calculate statistics efficiently
        # This approach keeps data in DuckDB and uses optimized window functions
        query = """
        WITH factor_stats AS (
            -- Calculate cross-sectional statistics by factor
            SELECT 
                fr.company_key,
                fr.date_key,
                fr.factor_key,
                fr.factor_value as raw_value,
                c.sector,
                -- Overall market statistics
                AVG(fr.factor_value) OVER (PARTITION BY fr.date_key, fr.factor_key) as factor_mean,
                STDDEV(fr.factor_value) OVER (PARTITION BY fr.date_key, fr.factor_key) as factor_std,
                COUNT(*) OVER (PARTITION BY fr.date_key, fr.factor_key) as factor_count,
                -- Percentile rank (0 to 1)
                PERCENT_RANK() OVER (
                    PARTITION BY fr.date_key, fr.factor_key 
                    ORDER BY fr.factor_value
                ) as percentile_rank,
                -- Sector statistics (Phase 2 - placeholder for now)
                NULL::DOUBLE as sector_mean,
                NULL::DOUBLE as sector_std
            FROM ANALYTICS.FACT_FACTOR_RAW fr
            JOIN ANALYTICS.DIM_COMPANY c ON fr.company_key = c.company_key
            JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
            WHERE d.date = %(date)s
              AND fr.factor_value IS NOT NULL
              AND c.is_current = TRUE
        )
        SELECT 
            company_key,
            date_key,
            factor_key,
            raw_value,
            -- Calculate z-score with safety check for zero std
            CASE 
                WHEN factor_std > 0 THEN (raw_value - factor_mean) / factor_std
                ELSE 0
            END as z_score,
            percentile_rank,
            -- Sector z-score (Phase 2)
            NULL::DOUBLE as sector_z_score,
            -- Sector percentile (Phase 2)  
            NULL::DOUBLE as sector_percentile_rank,
            factor_count
        FROM factor_stats
        WHERE factor_count >= 2  -- Need at least 2 values for meaningful stats
        """
        
        analytics_data = self.database.fetch_all(query, {'date': self.processing_date})
        
        # Track statistics by factor
        factor_stats = {}
        for row in analytics_data:
            factor_key = row['FACTOR_KEY']
            if factor_key not in factor_stats:
                factor_stats[factor_key] = {
                    'count': 0,
                    'z_scores': [],
                    'percentiles': []
                }
            factor_stats[factor_key]['count'] += 1
            factor_stats[factor_key]['z_scores'].append(row['Z_SCORE'])
            factor_stats[factor_key]['percentiles'].append(row['PERCENTILE_RANK'])
        
        # Log summary statistics
        for factor_key, stats in factor_stats.items():
            z_scores = stats['z_scores']
            if z_scores:
                mean_z = sum(z_scores) / len(z_scores)
                logger.info(f"Factor {factor_key}: {stats['count']} values, "
                           f"mean z-score: {mean_z:.4f}")
        
        # Store metadata
        self.result.metadata['factor_stats'] = factor_stats
        self.result.metadata['analytics_records'] = len(analytics_data)
        
        logger.info(f"Calculated statistics for {len(analytics_data)} factor values")
        
        return analytics_data
    
    def load(self, transformed_data: List[Dict[str, Any]]) -> int:
        """
        Load calculated statistics to FACT_FACTOR_ANALYTICS
        Uses DELETE + INSERT pattern for consistency
        
        Args:
            transformed_data: List of records with calculated statistics
            
        Returns:
            Number of records loaded
        """
        if not transformed_data:
            logger.warning("No data to load")
            return 0
        
        # Add timestamps
        current_timestamp = datetime.now(timezone.utc)
        for record in transformed_data:
            record['calculation_timestamp'] = current_timestamp
        
        # Use manual transaction for atomic DELETE + INSERT
        try:
            # Begin transaction
            self.database.execute("BEGIN")
            
            try:
                # Delete existing analytics for this date
                delete_query = """
                DELETE FROM ANALYTICS.FACT_FACTOR_ANALYTICS
                WHERE date_key IN (
                    SELECT date_key 
                    FROM ANALYTICS.DIM_DATE 
                    WHERE date = %(date)s
                )
                """
                
                deleted = self.database.execute(delete_query, {'date': self.processing_date})
                logger.info(f"Deleted {deleted} existing analytics records for {self.processing_date}")
                
                # Prepare data for bulk insert
                insert_data = []
                for row in transformed_data:
                    insert_data.append({
                        'company_key': row['COMPANY_KEY'],
                        'date_key': row['DATE_KEY'],
                        'factor_key': row['FACTOR_KEY'],
                        'raw_value': float(row['RAW_VALUE']),
                        'z_score': float(row['Z_SCORE']),
                        'percentile_rank': float(row['PERCENTILE_RANK']),
                        'sector_z_score': row['SECTOR_Z_SCORE'],  # NULL for now
                        'sector_percentile_rank': row['SECTOR_PERCENTILE_RANK'],  # NULL for now
                        'calculation_timestamp': current_timestamp
                    })
                
                # Bulk insert new analytics data
                self.database.bulk_insert(
                    "ANALYTICS.FACT_FACTOR_ANALYTICS",
                    insert_data
                )
                
                # Commit transaction
                self.database.execute("COMMIT")
                
                logger.info(f"Successfully loaded {len(insert_data)} analytics records")
                
            except Exception as e:
                # Rollback on error
                self.database.execute("ROLLBACK")
                raise
                
        except Exception as e:
            error_msg = f"Failed to load analytics data: {str(e)}"
            logger.error(error_msg)
            self.result.errors.append(error_msg)
            raise
        
        return len(insert_data)
    
    def run(self, calculation_date: str) -> Dict[str, Any]:
        """
        Run the factor analytics ETL pipeline for a specific date
        
        Args:
            calculation_date: Date to process (YYYY-MM-DD format)
            
        Returns:
            Result dictionary with status and statistics
        """
        self.reset_result()
        logger.info(f"Starting factor analytics ETL for {calculation_date}")
        
        try:
            # Extract raw factor values
            raw_data = self.extract(calculation_date)
            self.result.records_extracted = len(raw_data)
            
            if not raw_data:
                logger.warning(f"No data to process for {calculation_date}")
                self.result.status = 'SUCCESS'
                return {
                    'status': 'SUCCESS',
                    'message': f'No raw factor data found for {calculation_date}',
                    'records_extracted': 0,
                    'records_transformed': 0,
                    'records_loaded': 0,
                    'errors': [],
                    'factor_stats': {}
                }
            
            # Transform - calculate statistics
            analytics_data = self.transform(raw_data)
            self.result.records_transformed = len(analytics_data)
            
            # Load to analytics table
            records_loaded = self.load(analytics_data)
            self.result.records_loaded = records_loaded
            
            # Determine status
            if self.result.errors:
                self.result.status = 'PARTIAL'
            else:
                self.result.status = 'SUCCESS'
            
            return {
                'status': self.result.status,
                'records_extracted': self.result.records_extracted,
                'records_transformed': self.result.records_transformed,
                'records_loaded': self.result.records_loaded,
                'errors': self.result.errors,
                'factor_stats': self.result.metadata.get('factor_stats', {})
            }
            
        except Exception as e:
            logger.error(f"Factor analytics ETL failed: {str(e)}")
            self.result.status = 'FAILED'
            self.result.errors.append(str(e))
            raise
        
        finally:
            self.result.end_time = datetime.now(timezone.utc)
            duration = (self.result.end_time - self.result.start_time).total_seconds()
            logger.info(f"Analytics ETL completed in {duration:.2f} seconds "
                       f"with status: {self.result.status}")
    
    def process_date_range(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Process analytics for a range of dates
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Summary of processing results
        """
        # Get all dates with raw data in the range
        query = """
        SELECT DISTINCT d.date
        FROM ANALYTICS.FACT_FACTOR_RAW fr
        JOIN ANALYTICS.DIM_DATE d ON fr.date_key = d.date_key
        WHERE d.date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY d.date
        """
        
        dates = self.database.fetch_all(query, {
            'start_date': start_date,
            'end_date': end_date
        })
        
        results = []
        for date_row in dates:
            date = date_row['DATE']
            logger.info(f"Processing analytics for {date}")
            
            try:
                result = self.run(str(date))
                results.append({
                    'date': str(date),
                    'status': result['status'],
                    'records': result['records_loaded']
                })
            except Exception as e:
                logger.error(f"Failed to process {date}: {str(e)}")
                results.append({
                    'date': str(date),
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        # Summary
        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        total_records = sum(r.get('records', 0) for r in results)
        
        return {
            'dates_processed': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'total_records': total_records,
            'details': results
        }