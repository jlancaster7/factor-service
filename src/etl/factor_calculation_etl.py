"""
Factor Calculation ETL Pipeline
Inherits from BaseETL to calculate equity factors and load to staging
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from loguru import logger

from src.etl.base_etl import BaseETL
from src.data.snowflake_connector import SnowflakeConnector
from src.data.data_loader import DataLoader
from src.factors.registry import FactorRegistry
from src.config import Config

# Import factors to register them
import src.factors.momentum
import src.factors.value  
import src.factors.technical


class FactorCalculationETL(BaseETL):
    """ETL pipeline for calculating and loading equity factors"""
    
    def __init__(self, config: Config):
        """
        Initialize Factor Calculation ETL
        
        Args:
            config: Application configuration object
        """
        snowflake_config = config.get_snowflake_config()
        snowflake_connector = SnowflakeConnector(snowflake_config)
        snowflake_connector.connect()  # Ensure connection is established
        
        super().__init__(
            job_name="factor_calculation_etl",
            snowflake_connector=snowflake_connector,
            fmp_client=None,  # Not needed for factors
            batch_size=config.batch_size or 1000,
            max_retries=3,
            retry_delay=5,
            enable_monitoring=False  # Initially disabled
        )
        
        self.config = config
        self.data_loader = DataLoader(snowflake_connector)
        self.calculation_date = None
        self.symbols = None
        
    def extract(self, symbols: Optional[List[str]] = None, 
                start_date: Optional[str] = None, 
                end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract data required for factor calculations
        
        Args:
            symbols: List of stock symbols (None = all active)
            start_date: Start date for calculations
            end_date: End date for calculations
            
        Returns:
            Dictionary with 'price' and 'fundamental' DataFrames
        """
        # Set defaults
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = end_date  # Single day by default
            
        self.calculation_date = end_date
        
        # Get symbols if not provided
        if not symbols:
            symbols = self._get_active_symbols(end_date)
        self.symbols = symbols
        
        logger.info(f"Extracting data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Store metadata
        self.result.metadata['symbols'] = symbols
        self.result.metadata['start_date'] = start_date
        self.result.metadata['end_date'] = end_date
        
        # Determine lookback period based on registered factors
        max_lookback = self._get_max_lookback_days()
        logger.info(f"Using maximum lookback of {max_lookback} days for factor calculations")
        
        # Calculate start date for lookback period
        from datetime import datetime, timedelta
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_obj = end_date_obj - timedelta(days=max_lookback)
        lookback_start_date = start_date_obj.strftime("%Y-%m-%d")
        
        # Load price data with lookback using existing method
        price_data = self.data_loader.load_price_data(
            symbols=symbols,
            start_date=lookback_start_date,
            end_date=end_date
        )
        
        # Load fundamental data (point-in-time as of end_date)
        fundamental_data = self.data_loader.load_fundamental_data(
            symbols=symbols,
            as_of_date=end_date
        )
        
        # Load market metrics for convenience
        market_data = self.data_loader.load_market_metrics(
            symbols=symbols,
            date=end_date
        )
        
        logger.info(f"Extracted {len(price_data)} price records, "
                   f"{len(fundamental_data)} fundamental records, "
                   f"{len(market_data)} market metric records")
        
        return {
            'price': price_data,
            'fundamental': fundamental_data,
            'market': market_data
        }
    
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate all registered factors
        
        Args:
            raw_data: Dictionary with price, fundamental, and market DataFrames
            
        Returns:
            Dictionary with 'staging' key containing factor values
        """
        price_data = raw_data.get('price', pd.DataFrame())
        fundamental_data = raw_data.get('fundamental', pd.DataFrame())
        market_data = raw_data.get('market', pd.DataFrame())
        
        staging_records = []
        factor_stats = {}
        
        # Get all registered factors
        all_factors = FactorRegistry.list_factors()
        logger.info(f"Calculating {len(all_factors)} factors")
        
        for factor_name in all_factors:
            try:
                # Get factor instance
                factor_class = FactorRegistry.get(factor_name)
                factor = factor_class()
                
                logger.info(f"Calculating {factor_name} ({factor.category})")
                
                # Select appropriate data based on factor category
                if factor.category in ['momentum', 'technical']:
                    data = price_data
                elif factor.category == 'value':
                    # For book_to_market, we can use market metrics directly
                    if factor_name == 'book_to_market':
                        data = market_data
                    else:
                        data = fundamental_data
                else:
                    # Default to fundamental data
                    data = fundamental_data
                
                # Calculate factor values
                factor_values, diagnostics = factor.calculate_with_diagnostics(data)
                
                # Store diagnostics
                factor_stats[factor_name] = {
                    'calculated': len(factor_values),
                    'nulls': diagnostics['null_count'],
                    'infs': diagnostics['inf_count'],
                    'mean': diagnostics['mean'],
                    'std': diagnostics['std']
                }
                
                # Convert to staging records
                for symbol, value in factor_values.items():
                    if pd.notna(value) and not np.isinf(value):
                        staging_records.append({
                            'symbol': symbol,
                            'date': self.calculation_date,
                            'factor_name': factor_name,
                            'factor_value': float(value)
                        })
                
                logger.info(f"Calculated {factor_name}: "
                           f"{len(factor_values)} values, "
                           f"{diagnostics['null_count']} nulls, "
                           f"{diagnostics['inf_count']} infs")
                
            except Exception as e:
                error_msg = f"Failed to calculate {factor_name}: {str(e)}"
                logger.error(error_msg)
                self.result.errors.append(error_msg)
                
                # Store error in stats
                factor_stats[factor_name] = {
                    'calculated': 0,
                    'error': str(e)
                }
                
                # Continue with other factors (graceful failure)
                continue
        
        # Store factor statistics in metadata
        self.result.metadata['factor_stats'] = factor_stats
        self.result.metadata['total_factor_values'] = len(staging_records)
        
        logger.info(f"Transformed {len(staging_records)} factor values across all factors")
        
        return {'staging': staging_records}
    
    def load(self, transformed_data: Dict[str, List[Dict[str, Any]]]) -> int:
        """
        Load factor values to staging table using MERGE
        
        Args:
            transformed_data: Dictionary with staging records
            
        Returns:
            Number of records loaded
        """
        staging_data = transformed_data.get('staging', [])
        
        if not staging_data:
            logger.warning("No staging data to load")
            return 0
        
        # Add calculation timestamp
        current_timestamp = datetime.now(timezone.utc)
        for record in staging_data:
            record['calculation_timestamp'] = current_timestamp
        
        logger.info(f"Loading {len(staging_data)} records to STAGING.STG_FACTOR_VALUES")
        
        # Use MERGE for idempotent loads
        merge_keys = ['symbol', 'date', 'factor_name']
        update_columns = ['factor_value', 'calculation_timestamp']
        
        try:
            records_merged = self.snowflake.merge(
                table="STAGING.STG_FACTOR_VALUES",
                data=staging_data,
                merge_keys=merge_keys,
                update_columns=update_columns
            )
            
            logger.info(f"Successfully merged {records_merged} records to staging")
            return len(staging_data)
            
        except Exception as e:
            error_msg = f"Failed to load staging data: {str(e)}"
            logger.error(error_msg)
            self.result.errors.append(error_msg)
            raise
    
    def run(self, symbols: Optional[List[str]] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the factor calculation ETL pipeline
        
        Args:
            symbols: List of symbols to process (None = all active)
            start_date: Start date for calculation (supports backfill)
            end_date: End date for calculation (default = today)
            
        Returns:
            Result dictionary with status and statistics
        """
        logger.info(f"Starting factor calculation ETL")
        
        try:
            # Support single date or date range
            if start_date and not end_date:
                end_date = start_date
            
            # Extract data
            raw_data = self.extract(symbols, start_date, end_date)
            self.result.records_extracted = sum(len(df) for df in raw_data.values())
            
            # Transform - calculate factors
            transformed_data = self.transform(raw_data)
            self.result.records_transformed = len(transformed_data.get('staging', []))
            
            # Load to staging
            records_loaded = self.load(transformed_data)
            self.result.records_loaded = records_loaded
            
            # Log calculation summary
            self._log_calculation_summary()
            
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
            logger.error(f"Factor calculation ETL failed: {str(e)}")
            self.result.status = 'FAILED'
            self.result.errors.append(str(e))
            raise
        
        finally:
            self.result.end_time = datetime.now(timezone.utc)
            duration = (self.result.end_time - self.result.start_time).total_seconds()
            logger.info(f"ETL completed in {duration:.2f} seconds with status: {self.result.status}")
    
    def _get_active_symbols(self, as_of_date: str) -> List[str]:
        """Get list of active symbols to process"""
        query = """
        SELECT DISTINCT symbol 
        FROM ANALYTICS.DIM_COMPANY 
        WHERE is_current = TRUE
        ORDER BY symbol
        """
        
        result = self.snowflake.fetch_all(query)
        symbols = [row['SYMBOL'] for row in result]
        
        logger.info(f"Found {len(symbols)} active symbols")
        return symbols
    
    def _get_max_lookback_days(self) -> int:
        """Determine maximum lookback period from registered factors"""
        max_lookback = 252  # Default to 1 year
        
        for factor_name in FactorRegistry.list_factors():
            try:
                factor_class = FactorRegistry.get(factor_name)
                factor = factor_class()
                
                if hasattr(factor, 'lookback_days'):
                    max_lookback = max(max_lookback, factor.lookback_days)
            except:
                pass
        
        return max_lookback
    
    def _log_calculation_summary(self):
        """Log summary to ANALYTICS.FACTOR_CALCULATION_LOG"""
        factor_stats = self.result.metadata.get('factor_stats', {})
        
        log_entries = []
        for factor_name, stats in factor_stats.items():
            log_entry = {
                'calculation_date': self.calculation_date,
                'factor_name': factor_name,
                'symbols_processed': len(self.symbols) if self.symbols else 0,
                'records_calculated': stats.get('calculated', 0),
                'null_count': stats.get('nulls', 0),
                'inf_count': stats.get('infs', 0),
                'calculation_time_seconds': 0,  # Would need per-factor timing
                'status': 'ERROR' if 'error' in stats else 'SUCCESS',
                'error_message': stats.get('error'),
                'created_timestamp': datetime.now(timezone.utc)
            }
            log_entries.append(log_entry)
        
        if log_entries:
            try:
                self.snowflake.bulk_insert(
                    "ANALYTICS.FACTOR_CALCULATION_LOG", 
                    log_entries
                )
                logger.info(f"Logged {len(log_entries)} calculation results")
            except Exception as e:
                logger.warning(f"Failed to log calculation results: {e}")