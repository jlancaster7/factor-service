"""Technical indicator factor implementations."""
import pandas as pd
import numpy as np
from typing import List
from .base import BaseFactor
from .registry import FactorRegistry
import logging

logger = logging.getLogger(__name__)


@FactorRegistry.register
class RSI14(BaseFactor):
    """14-day Relative Strength Index (RSI) technical indicator.
    
    RSI measures momentum by comparing the magnitude of recent gains to recent losses.
    Values range from 0 to 100:
    - RSI > 70: Potentially overbought
    - RSI < 30: Potentially oversold
    - RSI = 50: Neutral momentum
    
    Formula:
    RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain / Average Loss over the period
    """
    
    def __init__(self):
        super().__init__(name="rsi_14", category="technical")
        self.period = 14  # Standard RSI period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate 14-day RSI for each symbol.
        
        Args:
            data: DataFrame with columns [symbol, date, adj_close]
            
        Returns:
            Series indexed by symbol with RSI values
        """
        self.validate_data(data)
        
        results = {}
        
        # Process each symbol
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            # Need at least period + 1 days for calculation
            if len(symbol_data) <= self.period:
                logger.warning(f"{self.name}: Insufficient data for {symbol} "
                             f"({len(symbol_data)} days, need {self.period + 1})")
                results[symbol] = np.nan
                continue
            
            try:
                # Get price series
                prices = symbol_data['adj_close'].values
                
                # Calculate price changes
                deltas = np.diff(prices)
                
                # Separate gains and losses
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                # Calculate RSI using Wilder's smoothing method
                # First average uses simple mean
                avg_gain = np.mean(gains[:self.period])
                avg_loss = np.mean(losses[:self.period])
                
                # Apply Wilder's smoothing for subsequent values
                for i in range(self.period, len(gains)):
                    avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
                    avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period
                
                # Calculate RSI
                if avg_loss == 0:
                    # All gains, no losses - RSI = 100
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                results[symbol] = rsi
                
                logger.debug(f"{self.name}: {symbol} RSI = {rsi:.2f} "
                           f"(avg_gain: {avg_gain:.4f}, avg_loss: {avg_loss:.4f})")
                
            except Exception as e:
                logger.error(f"{self.name}: Error calculating RSI for {symbol}: {e}")
                results[symbol] = np.nan
        
        return pd.Series(results, name=self.name)
    
    def get_required_columns(self) -> List[str]:
        """Return required columns for RSI calculation."""
        return ['symbol', 'date', 'adj_close']
    
    def calculate_simple(self, data: pd.DataFrame) -> pd.Series:
        """Alternative simple RSI calculation (for comparison).
        
        This uses a simple moving average instead of Wilder's smoothing.
        """
        self.validate_data(data)
        
        results = {}
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            
            if len(symbol_data) <= self.period:
                results[symbol] = np.nan
                continue
            
            try:
                # Get recent prices (last period + 1 days)
                recent_prices = symbol_data['adj_close'].tail(self.period + 1).values
                
                # Calculate changes
                changes = np.diff(recent_prices)
                
                # Calculate average gain and loss
                gains = changes[changes > 0]
                losses = -changes[changes < 0]
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                # Calculate RSI
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                results[symbol] = rsi
                
            except Exception as e:
                logger.error(f"{self.name}: Error in simple RSI for {symbol}: {e}")
                results[symbol] = np.nan
        
        return pd.Series(results, name=f"{self.name}_simple")