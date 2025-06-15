"""Data loading utilities that convert Snowflake results to DataFrames."""
import pandas as pd
from typing import List, Dict, Any, Optional
from decimal import Decimal
from loguru import logger

from .snowflake_connector import SnowflakeConnector


class DataLoader:
    """Helper class to load data from Snowflake and convert to DataFrames."""
    
    def __init__(self, connector: SnowflakeConnector):
        """Initialize with a Snowflake connector."""
        self.connector = connector
    
    def query_to_dataframe(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            params: Optional query parameters
            
        Returns:
            pandas DataFrame with query results
        """
        # Get results as List[Dict]
        results = self.connector.fetch_all(query, params)
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            logger.debug(f"Converted {len(results)} rows to DataFrame with columns: {df.columns.tolist()}")
            
            # Convert Decimal columns to float to avoid type issues
            df = self._convert_decimal_columns(df)
            
            return df
        else:
            logger.warning("Query returned no results")
            return pd.DataFrame()
    
    def _convert_decimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Decimal type columns to float.
        
        Snowflake returns Decimal types for numeric columns, which can cause
        issues with pandas operations. This method converts them to float.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with Decimal columns converted to float
        """
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                # Check if first non-null value is Decimal
                non_null = df[col].dropna()
                if not non_null.empty and isinstance(non_null.iloc[0], Decimal):
                    logger.debug(f"Converting column '{col}' from Decimal to float")
                    df[col] = df[col].astype(float)
        
        return df
    
    def load_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load daily price data for given symbols and date range.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        query = """
        SELECT 
            c.symbol,
            d.date,
            p.adj_close,
            p.close_price,
            p.volume,
            p.change_percent
        FROM ANALYTICS.FACT_DAILY_PRICES p
        JOIN ANALYTICS.DIM_COMPANY c ON p.company_key = c.company_key
        JOIN ANALYTICS.DIM_DATE d ON p.date_key = d.date_key
        WHERE c.symbol IN ({placeholders})
          AND d.date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY c.symbol, d.date
        """
        
        # Create placeholders for IN clause
        placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(symbols))])
        query = query.format(placeholders=placeholders)
        
        # Build parameters
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        for i, symbol in enumerate(symbols):
            params[f'symbol_{i}'] = symbol
        
        return self.query_to_dataframe(query, params)
    
    def load_fundamental_data(self, symbols: List[str], as_of_date: str) -> pd.DataFrame:
        """
        Load point-in-time fundamental data for given symbols.
        
        Args:
            symbols: List of stock symbols
            as_of_date: Point-in-time date (YYYY-MM-DD)
            
        Returns:
            DataFrame with fundamental data
        """
        query = """
        WITH latest_financials AS (
            SELECT 
                f.*,
                c.symbol,
                d.date as fiscal_date,
                ROW_NUMBER() OVER (
                    PARTITION BY f.company_key 
                    ORDER BY f.accepted_date DESC
                ) as rn
            FROM ANALYTICS.FACT_FINANCIALS f
            JOIN ANALYTICS.DIM_COMPANY c ON f.company_key = c.company_key
            JOIN ANALYTICS.DIM_DATE d ON f.fiscal_date_key = d.date_key
            WHERE c.symbol IN ({placeholders})
              AND f.accepted_date <= %(as_of_date)s
        )
        SELECT * FROM latest_financials WHERE rn = 1
        """
        
        # Create placeholders for IN clause
        placeholders = ','.join([f"%(symbol_{i})s" for i in range(len(symbols))])
        query = query.format(placeholders=placeholders)
        
        # Build parameters
        params = {'as_of_date': as_of_date}
        for i, symbol in enumerate(symbols):
            params[f'symbol_{i}'] = symbol
        
        return self.query_to_dataframe(query, params)