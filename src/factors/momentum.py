"""Momentum-based factor implementations."""
import pandas as pd
import numpy as np
from typing import List
from .base import BaseFactor
from .registry import FactorRegistry
import logging

logger = logging.getLogger(__name__)


@FactorRegistry.register
class Momentum12_1(BaseFactor):
    """12-month minus 1-month momentum factor.
    
    Calculates the return from 12 months ago to 1 month ago.
    This skips the most recent month to avoid short-term reversal effects.
    
    Formula: (Price_t-21 / Price_t-252) - 1
    Where t-21 is approximately 1 month ago and t-252 is approximately 12 months ago.
    """
    
    def __init__(self):
        super().__init__(name="momentum_12_1", category="momentum")
        self.lookback_days = 252  # ~12 months of trading days
        self.skip_days = 21      # ~1 month of trading days
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate 12-1 momentum for each symbol.
        
        Args:
            data: DataFrame with columns [symbol, date, adj_close]
            
        Returns:
            Series indexed by symbol with momentum values
        """
        self.validate_data(data)
        
        results = {}
        
        # Process each symbol separately
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            # Need at least lookback_days of history
            if len(symbol_data) < self.lookback_days:
                logger.warning(f"{self.name}: Insufficient data for {symbol} "
                             f"({len(symbol_data)} days, need {self.lookback_days})")
                results[symbol] = np.nan
                continue
            
            try:
                # Get the most recent date's data
                current_idx = -1
                
                # Price from skip_days ago (1 month)
                skip_idx = current_idx - self.skip_days
                if abs(skip_idx) > len(symbol_data):
                    logger.warning(f"{self.name}: Cannot calculate skip price for {symbol}")
                    results[symbol] = np.nan
                    continue
                
                # Price from lookback_days ago (12 months)
                lookback_idx = current_idx - self.lookback_days
                if abs(lookback_idx) > len(symbol_data):
                    logger.warning(f"{self.name}: Cannot calculate lookback price for {symbol}")
                    results[symbol] = np.nan
                    continue
                
                skip_price = symbol_data['adj_close'].iloc[skip_idx]
                lookback_price = symbol_data['adj_close'].iloc[lookback_idx]
                
                # Check for valid prices
                if pd.isna(skip_price) or pd.isna(lookback_price) or lookback_price <= 0:
                    logger.warning(f"{self.name}: Invalid prices for {symbol} "
                                 f"(skip: {skip_price}, lookback: {lookback_price})")
                    results[symbol] = np.nan
                    continue
                
                # Calculate momentum
                momentum = (skip_price / lookback_price) - 1
                results[symbol] = momentum
                
                logger.debug(f"{self.name}: {symbol} momentum = {momentum:.4f} "
                           f"(skip_price: {skip_price:.2f}, lookback_price: {lookback_price:.2f})")
                
            except Exception as e:
                logger.error(f"{self.name}: Error calculating momentum for {symbol}: {e}")
                results[symbol] = np.nan
        
        return pd.Series(results, name=self.name)
    
    def get_required_columns(self) -> List[str]:
        """Return required columns for momentum calculation."""
        return ['symbol', 'date', 'adj_close']