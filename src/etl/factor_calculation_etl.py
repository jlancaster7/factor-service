"""
Factor Calculation ETL Pipeline V2 - Refactored for fact tables
Inherits from BaseETL to calculate equity factors and load to fact tables
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.etl.base_etl import BaseETL
from src.data.database import DatabaseConnector
from src.data.data_loader import DataLoader
from src.factors.registry import FactorRegistry
from src.config import Config

# Import factors to register them
import src.factors.momentum
import src.factors.value  
import src.factors.technical


class FactorCalculationETL(BaseETL):
    """ETL pipeline for calculating and loading equity factors to fact tables"""
    
    def __init__(self, config: Config):
        """
        Initialize Factor Calculation ETL
        
        Args:
            config: Application configuration object
        """
        database_connector = DatabaseConnector(config)
        database_connector.initialize_schema()  # Ensure connection is established
        
        super().__init__(
            job_name="factor_calculation_etl",
            database_connector=database_connector,
            fmp_client=None,  # Not needed for factors
            batch_size=config.batch_size or 1000,
            max_retries=3,
            retry_delay=5,
            enable_monitoring=False  # Initially disabled
        )
        
        self.config = config
        self.data_loader = DataLoader(database_connector)
        self.calculation_dates = []  # List of dates to process
        self.symbols = None
        
        # Cache for dimension lookups
        self._company_key_cache = {}
        self._date_key_cache = {}
        self._factor_key_cache = {}
        
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
            
        # Get all dates in range for calculations
        dates_query = """
        SELECT DISTINCT date
        FROM ANALYTICS.DIM_DATE d
        WHERE EXISTS (
            SELECT 1 FROM ANALYTICS.FACT_DAILY_PRICES p
            WHERE p.date_key = d.date_key
        ) AND date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY date
        """
        date_results = self.database.fetch_all(dates_query, {'start_date': start_date, 'end_date': end_date})
        self.calculation_dates = [str(row['DATE']) for row in date_results]
        
        # Get symbols if not provided
        if not symbols:
            symbols = self._get_active_symbols(end_date)
        if not symbols:
            raise ValueError("No companies loaded. Import local source data before running factors.")
        self.symbols = symbols
        
        logger.info(f"Extracting data for {len(symbols)} symbols for {len(self.calculation_dates)} dates "
                   f"({start_date} to {end_date})")

        # Load dimension keys for the symbols and date range
        self._load_dimension_keys(symbols, start_date, end_date)
        
        # Store metadata
        self.result.metadata['symbols'] = symbols
        self.result.metadata['start_date'] = start_date
        self.result.metadata['end_date'] = end_date
        
        # Determine lookback period based on registered factors
        max_lookback_calendar_days = self._get_max_lookback_days()
        logger.info(f"Using lookback of {max_lookback_calendar_days} calendar days to ensure sufficient trading days")
        
        # Calculate start date for lookback period
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=max_lookback_calendar_days)
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
        
        # Load market metrics for date range
        market_data = self.data_loader.load_market_metrics(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
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
            Dictionary with 'staging' and 'fact_raw' keys containing factor values
        """
        price_data = raw_data.get('price', pd.DataFrame())
        fundamental_data = raw_data.get('fundamental', pd.DataFrame())
        market_data = raw_data.get('market', pd.DataFrame())
        
        fact_raw_records = []  # Only fact table now
        factor_stats = {}
        
        # Get all registered factors
        all_factors = FactorRegistry.list_factors()
        logger.info(f"Calculating {len(all_factors)} factors for {len(self.calculation_dates)} dates")

        # Process each date
        for calculation_date in self.calculation_dates:
            logger.info(f"Processing date: {calculation_date}")

            for factor_name in all_factors:
                try:
                    # Get factor instance
                    factor_class = FactorRegistry.get(factor_name)
                    factor = factor_class()

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

                    if factor.category in ['momentum', 'technical']:
                        day = datetime.strptime(calculation_date, "%Y-%m-%d").date()
                        observed = set(data.loc[data['date'] == day, 'symbol']) if not data.empty else set()
                        data = data[data['symbol'].isin(observed)] if not data.empty else data
                    if data.empty:
                        logger.warning(f"No input data for {factor_name} on {calculation_date}")
                        continue

                    # Calculate factor values for this specific date
                    factor_values, diagnostics = factor.calculate_with_diagnostics(data, calculation_date)

                    # Update cumulative diagnostics
                    if factor_name not in factor_stats:
                        factor_stats[factor_name] = {
                            'dates_calculated': 0,
                            'total_values': 0,
                            'total_nulls': 0,
                            'total_infs': 0
                        }

                    factor_stats[factor_name]['dates_calculated'] += 1
                    factor_stats[factor_name]['total_values'] += len(factor_values)
                    factor_stats[factor_name]['total_nulls'] += diagnostics['null_count']
                    factor_stats[factor_name]['total_infs'] += diagnostics['inf_count']

                    # Get factor key
                    factor_key = self._factor_key_cache.get(factor_name)
                    if not factor_key:
                        logger.warning(f"No factor key found for {factor_name}")
                        continue

                    # Convert to records (fact table only, no more staging)
                    for symbol, value in factor_values.items():
                        if pd.notna(value) and not np.isinf(value):
                            company_key = self._company_key_cache.get(symbol)
                            date_key = self._date_key_cache.get(calculation_date)

                            if company_key and date_key:
                                fact_raw_records.append({
                                    'company_key': company_key,
                                    'date_key': date_key,
                                    'factor_key': factor_key,
                                    'factor_value': float(value)
                                })
                            else:
                                logger.warning(f"Missing dimension keys for {symbol} on {calculation_date}")

                    logger.debug(f"Date {calculation_date} - {factor_name}: "
                               f"{len(factor_values)} values, "
                               f"{diagnostics['null_count']} nulls")

                except Exception as e:
                    error_msg = f"Failed to calculate {factor_name} for {calculation_date}: {str(e)}"
                    logger.error(error_msg)
                    self.result.errors.append(error_msg)

                    # Continue with other factors (graceful failure)
                    continue
        
        # Store factor statistics in metadata
        self.result.metadata['factor_stats'] = factor_stats
        self.result.metadata['total_fact_values'] = len(fact_raw_records)
        
        logger.info(f"Transformed {len(fact_raw_records)} fact records")
        
        return {
            'fact_raw': fact_raw_records
        }
    
    def load(self, transformed_data: Dict[str, List[Dict[str, Any]]]) -> int:
        """
        Load factor values to fact table only
        
        Args:
            transformed_data: Dictionary with fact_raw records
            
        Returns:
            Number of records loaded
        """
        fact_data = transformed_data.get('fact_raw', [])
        
        if not fact_data:
            logger.warning("No data to load")
            return 0
        
        # Add calculation timestamp
        current_timestamp = datetime.now(timezone.utc)
        
        # Load to fact table only
        if fact_data:
            for record in fact_data:
                record['calculation_timestamp'] = current_timestamp
            
            logger.info(f"Loading {len(fact_data)} records to ANALYTICS.FACT_FACTOR_RAW")
            
        try:
            # Use MERGE for idempotent loads
            merge_keys = ['company_key', 'date_key', 'factor_key']
            update_columns = ['factor_value', 'calculation_timestamp']
            
            records_merged = self.database.merge(
                table="ANALYTICS.FACT_FACTOR_RAW",
                data=fact_data,
                merge_keys=merge_keys,
                update_columns=update_columns
            )
            
            logger.info(f"Successfully merged {records_merged} records to fact table")
            return len(fact_data)
            
        except Exception as e:
            error_msg = f"Failed to load fact data: {str(e)}"
            logger.error(error_msg)
            self.result.errors.append(error_msg)
            return 0
    
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
        self.reset_result()
        logger.info("Starting factor calculation ETL")
        
        try:
            # Support single date or date range
            if start_date and not end_date:
                end_date = start_date
            
            # Extract data
            raw_data = self.extract(symbols, start_date, end_date)
            self.result.records_extracted = sum(len(df) for df in raw_data.values())
            
            # Transform - calculate factors
            transformed_data = self.transform(raw_data)
            self.result.records_transformed = len(transformed_data.get('fact_raw', []))
            
            # Load to fact table only
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
    
    def _load_dimension_keys(self, symbols: List[str], start_date: str, end_date: str):
        """Load dimension keys for symbols and date range"""
        # Load company keys
        query = """
        SELECT symbol, company_key 
        FROM ANALYTICS.DIM_COMPANY 
        WHERE symbol IN ({placeholders})
          AND is_current = TRUE
        """
        placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(symbols))])
        query = query.format(placeholders=placeholders)
        
        params = {}
        for i, symbol in enumerate(symbols):
            params[f'symbol_{i}'] = symbol
        
        results = self.database.fetch_all(query, params)
        self._company_key_cache = {row['SYMBOL']: row['COMPANY_KEY'] for row in results}
        
        # Load date keys for the range
        date_query = """
        SELECT date, date_key 
        FROM ANALYTICS.DIM_DATE 
        WHERE date BETWEEN %(start_date)s AND %(end_date)s
        """
        results = self.database.fetch_all(date_query, {'start_date': start_date, 'end_date': end_date})
        self._date_key_cache = {str(row['DATE']): row['DATE_KEY'] for row in results}
        
        # Load factor keys
        factor_query = """
        SELECT factor_name, factor_key 
        FROM ANALYTICS.DIM_FACTOR 
        WHERE is_active = TRUE
        """
        results = self.database.fetch_all(factor_query)
        self._factor_key_cache = {row['FACTOR_NAME']: row['FACTOR_KEY'] for row in results}
        
        logger.info(f"Loaded dimension keys: {len(self._company_key_cache)} companies, "
                   f"{len(self._date_key_cache)} dates, {len(self._factor_key_cache)} factors")
    
    def _get_active_symbols(self, as_of_date: str) -> List[str]:
        """Get list of active symbols to process"""
        query = """
        SELECT DISTINCT symbol 
        FROM ANALYTICS.DIM_COMPANY 
        WHERE is_current = TRUE
        ORDER BY symbol
        """
        
        result = self.database.fetch_all(query)
        symbols = [row['SYMBOL'] for row in result]
        
        logger.info(f"Found {len(symbols)} active symbols")
        return symbols
    
    def _get_max_lookback_days(self) -> int:
        """Determine maximum lookback period from registered factors"""
        max_lookback = 252  # Default to 1 year of trading days
        
        for factor_name in FactorRegistry.list_factors():
            try:
                factor_class = FactorRegistry.get(factor_name)
                factor = factor_class()
                
                if hasattr(factor, 'lookback_days'):
                    max_lookback = max(max_lookback, factor.lookback_days)
            except:
                pass
        
        # Convert trading days to calendar days (approximate)
        # 252 trading days ≈ 365 calendar days
        # Add some buffer to be safe
        calendar_days = int(max_lookback * 1.5)
        
        return calendar_days
    
    def _log_calculation_summary(self):
        """Log summary to ANALYTICS.FACTOR_CALCULATION_LOG"""
        factor_stats = self.result.metadata.get('factor_stats', {})
        
        log_entries = []
        # Log summary for the entire batch
        batch_end_date = self.calculation_dates[-1] if self.calculation_dates else None

        for factor_name, stats in factor_stats.items():
            log_entry = {
                'calculation_date': batch_end_date,  # Use last date as reference
                'factor_name': factor_name,
                'symbols_processed': len(self.symbols) if self.symbols else 0,
                'records_calculated': stats.get('total_values', 0),
                'null_count': stats.get('total_nulls', 0),
                'inf_count': stats.get('total_infs', 0),
                'calculation_time_seconds': 0,  # Would need per-factor timing
                'status': 'SUCCESS' if stats.get('dates_calculated', 0) > 0 else 'ERROR',
                'error_message': f"Processed {stats.get('dates_calculated', 0)} dates",
                'created_timestamp': datetime.now(timezone.utc)
            }
            log_entries.append(log_entry)
        
        if log_entries:
            try:
                self.database.bulk_insert(
                    "ANALYTICS.FACTOR_CALCULATION_LOG", 
                    log_entries
                )
                logger.info(f"Logged {len(log_entries)} calculation results")
            except Exception as e:
                logger.warning(f"Failed to log calculation results: {e}")