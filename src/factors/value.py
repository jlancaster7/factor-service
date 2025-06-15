"""Value-based factor implementations."""
import pandas as pd
import numpy as np
from typing import List
from .base import BaseFactor
from .registry import FactorRegistry
import logging

logger = logging.getLogger(__name__)


@FactorRegistry.register
class BookToMarket(BaseFactor):
    """Book value to market value ratio factor.
    
    Calculates the ratio of book value (total equity) to market capitalization.
    Higher values indicate potentially undervalued stocks.
    
    Formula: Total Equity / (Price * Shares Outstanding)
    
    This uses the most recent available fundamental data relative to the
    price date to ensure point-in-time accuracy.
    """
    
    def __init__(self):
        super().__init__(name="book_to_market", category="value")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate book-to-market ratio for each symbol.
        
        Book-to-Market = 1 / Price-to-Book
        
        Args:
            data: DataFrame with columns [symbol, date, pb_ratio]
                  From FACT_MARKET_METRICS table
            
        Returns:
            Series indexed by symbol with B/M values
        """
        self.validate_data(data)
        
        results = {}
        
        # Process each symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            
            # Sort by date and get the most recent data
            symbol_data = symbol_data.sort_values('date')
            
            # Find the most recent row with valid P/B ratio
            valid_data = symbol_data.dropna(subset=['pb_ratio'])
            
            if len(valid_data) == 0:
                logger.warning(f"{self.name}: No valid P/B data for {symbol}")
                results[symbol] = np.nan
                continue
            
            # Get the most recent valid data
            latest = valid_data.iloc[-1]
            
            try:
                pb_ratio = latest['pb_ratio']
                
                # Validate P/B ratio
                if pb_ratio <= 0:
                    logger.warning(f"{self.name}: Invalid P/B ratio for {symbol}: {pb_ratio}")
                    results[symbol] = np.nan
                    continue
                
                # Calculate B/M as inverse of P/B
                book_to_market = 1.0 / pb_ratio
                
                # Handle edge cases
                if np.isinf(book_to_market):
                    logger.warning(f"{self.name}: Infinite B/M for {symbol} (P/B: {pb_ratio})")
                    results[symbol] = np.nan
                else:
                    results[symbol] = book_to_market
                
                logger.debug(f"{self.name}: {symbol} B/M = {book_to_market:.4f} (P/B = {pb_ratio:.2f})")
                
            except Exception as e:
                logger.error(f"{self.name}: Error calculating B/M for {symbol}: {e}")
                results[symbol] = np.nan
        
        return pd.Series(results, name=self.name)
    
    def get_required_columns(self) -> List[str]:
        """Return required columns for B/M calculation."""
        return ['symbol', 'date', 'pb_ratio']