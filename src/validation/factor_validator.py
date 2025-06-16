"""
Factor data validation compatible with BaseETL
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from loguru import logger


class FactorDataValidator:
    """
    Validates data for BaseETL pipeline, with specialized logic for factor calculations.
    Implements the BaseETL-expected interface directly.
    """
    
    def __init__(self, 
                 max_inf_pct: float = 0.01,
                 max_nan_pct: float = 0.05,
                 outlier_threshold: float = 5.0):
        """
        Initialize validator with thresholds
        
        Args:
            max_inf_pct: Maximum percentage of infinite values allowed
            max_nan_pct: Maximum percentage of NaN values allowed
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.max_inf_pct = max_inf_pct
        self.max_nan_pct = max_nan_pct
        self.outlier_threshold = outlier_threshold
    
    def validate_batch(self, 
                      records: List[Dict[str, Any]], 
                      validation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate a batch of records (BaseETL-compatible interface)
        
        Args:
            records: List of records to validate
            validation_type: Type of validation (e.g., 'factor', 'price', etc.)
                           If None, will auto-detect based on record structure
            
        Returns:
            Validation results dictionary compatible with BaseETL
        """
        if not records:
            return self._empty_result()
        
        # Check if this is factor data
        if not self._is_factor_data(records, validation_type):
            # For non-factor data, return minimal validation
            return self._minimal_validation(records)
        
        # Perform factor-specific validation
        return self._validate_factor_data(records)
    
    def validate_factor_series(self, 
                             series: pd.Series, 
                             factor_name: str) -> Dict[str, Any]:
        """
        Validate a pandas Series of factor values
        
        Args:
            series: Series of factor values
            factor_name: Name of the factor
            
        Returns:
            Validation results dictionary
        """
        # Convert to records format
        records = [
            {
                "symbol": idx,
                "factor_value": value
            }
            for idx, value in series.items()
        ]
        
        return self.validate_batch(records, 'factor')
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty validation result"""
        return {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "issues": [],
            "issues_by_record": []
        }
    
    def _minimal_validation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Minimal validation for non-factor data"""
        return {
            "total_records": len(records),
            "valid_records": len(records),
            "invalid_records": 0,
            "issues": [],
            "issues_by_record": []
        }
    
    def _is_factor_data(self, records: List[Dict[str, Any]], 
                       validation_type: Optional[str] = None) -> bool:
        """Check if records contain factor data"""
        # Explicit type check
        if validation_type == 'factor':
            return True
        
        # Auto-detect based on record structure
        if not records:
            return False
        
        first_record = records[0]
        factor_fields = {'factor_name', 'factor_value'}
        return all(field in first_record for field in factor_fields)
    
    def _validate_factor_data(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform detailed factor validation"""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(records)
        
        # Extract factor name if available
        factor_name = records[0].get('factor_name', 'unknown') if records else 'unknown'
        
        # Extract factor values
        if 'factor_value' not in df.columns:
            return {
                "total_records": len(records),
                "valid_records": 0,
                "invalid_records": len(records),
                "issues": ["Missing factor_value column"],
                "issues_by_record": []
            }
        
        values = df['factor_value']
        total_records = len(values)
        
        # Calculate statistics
        nan_count = values.isna().sum()
        inf_count = np.isinf(values).sum()
        nan_pct = nan_count / total_records
        inf_pct = inf_count / total_records
        
        # Find outliers using z-score
        clean_values = values[~values.isna() & ~np.isinf(values)]
        if len(clean_values) > 0:
            mean = clean_values.mean()
            std = clean_values.std()
            if std > 0:
                z_scores = np.abs((clean_values - mean) / std)
                outlier_count = (z_scores > self.outlier_threshold).sum()
            else:
                outlier_count = 0
        else:
            outlier_count = 0
            mean = None
            std = None
        
        # Collect issues
        issues = []
        issues_by_record = []
        
        if nan_pct > self.max_nan_pct:
            issues.append(f"High NaN percentage: {nan_pct:.2%} (threshold: {self.max_nan_pct:.2%})")
        
        if inf_pct > self.max_inf_pct:
            issues.append(f"High Inf percentage: {inf_pct:.2%} (threshold: {self.max_inf_pct:.2%})")
        
        # Track individual record issues
        for idx, record in enumerate(records):
            record_issues = []
            value = record.get('factor_value')
            
            if value is None or pd.isna(value):
                record_issues.append("NaN value")
            elif np.isinf(value):
                record_issues.append("Infinite value")
            elif std and std > 0:
                z_score = abs((value - mean) / std)
                if z_score > self.outlier_threshold:
                    record_issues.append(f"Outlier (z-score: {z_score:.2f})")
            
            if record_issues:
                issues_by_record.append({
                    "record_identifier": f"{record.get('symbol', 'UNKNOWN')}_{record.get('date', 'UNKNOWN')}",
                    "issues": record_issues
                })
        
        invalid_records = len(issues_by_record)
        valid_records = total_records - invalid_records
        
        # Log summary
        if issues:
            logger.warning(f"Factor {factor_name} validation issues: {', '.join(issues)}")
        else:
            logger.info(f"Factor {factor_name} validation passed for {total_records} records")
        
        # Return BaseETL-compatible result
        result = {
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "issues": issues,
            "issues_by_record": issues_by_record[:10]  # Limit to first 10 for brevity
        }
        
        # Add statistics as metadata (BaseETL can use this if needed)
        result["statistics"] = {
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "outlier_count": int(outlier_count),
            "mean": float(mean) if mean is not None else None,
            "std": float(std) if std is not None else None,
            "min": float(clean_values.min()) if len(clean_values) > 0 else None,
            "max": float(clean_values.max()) if len(clean_values) > 0 else None
        }
        
        return result