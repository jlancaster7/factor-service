"""Base class for all factor calculations."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseFactor(ABC):
    """Base class for all factor calculations."""
    
    def __init__(self, name: str, category: str):
        """Initialize factor with name and category.
        
        Args:
            name: Unique factor name (e.g., 'momentum_12_1')
            category: Factor category (e.g., 'momentum', 'value', 'technical')
        """
        self.name = name
        self.category = category
        logger.info(f"Initialized factor: {name} (category: {category})")
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate factor values for all securities in the DataFrame.
        
        Args:
            data: DataFrame with required columns for calculation
            
        Returns:
            Series indexed by symbol with factor values
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of DataFrame columns required for calculation."""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that data contains required columns.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required = self.get_required_columns()
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns for {self.name}: {missing}")
    
    def calculate_with_diagnostics(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Calculate factor with diagnostic information.
        
        Args:
            data: DataFrame with required columns for calculation
            
        Returns:
            Tuple of (factor_values, diagnostics_dict)
        """
        # Validate data first
        self.validate_data(data)
        
        # Calculate factor values
        values = self.calculate(data)
        
        # Generate diagnostics
        diagnostics = {
            'factor_name': self.name,
            'category': self.category,
            'total_count': len(values),
            'null_count': values.isnull().sum(),
            'inf_count': np.isinf(values).sum() if pd.api.types.is_numeric_dtype(values) else 0,
            'zero_count': (values == 0).sum(),
            'unique_count': values.nunique(),
        }
        
        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(values) and len(values) > 0:
            valid_values = values[values.notna() & ~np.isinf(values)]
            if len(valid_values) > 0:
                diagnostics.update({
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'percentiles': {
                        '25%': float(valid_values.quantile(0.25)),
                        '50%': float(valid_values.quantile(0.50)),
                        '75%': float(valid_values.quantile(0.75))
                    }
                })
            else:
                diagnostics.update({
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'percentiles': {'25%': None, '50%': None, '75%': None}
                })
        
        # Log warnings for data quality issues
        if diagnostics['null_count'] > 0:
            null_pct = (diagnostics['null_count'] / diagnostics['total_count']) * 100
            logger.warning(f"{self.name}: {diagnostics['null_count']} null values found ({null_pct:.1f}%)")
        
        if diagnostics.get('inf_count', 0) > 0:
            logger.warning(f"{self.name}: {diagnostics['inf_count']} infinite values found")
        
        return values, diagnostics
    
    def __repr__(self) -> str:
        """String representation of the factor."""
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"