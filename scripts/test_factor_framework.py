#!/usr/bin/env python
"""Test the base factor framework."""
import sys
import logging
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

from src.factors.base import BaseFactor
from src.factors.registry import FactorRegistry


# Create a simple test factor for validation
@FactorRegistry.register
class TestFactor(BaseFactor):
    """Simple test factor for framework validation."""
    
    def __init__(self):
        super().__init__(name="test_factor", category="test")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Simple calculation: return closing price."""
        self.validate_data(data)
        
        # Group by symbol and get last close price
        result = data.groupby('symbol')['close'].last()
        return result.rename(self.name)
    
    def get_required_columns(self) -> list:
        return ['symbol', 'close']


def main():
    """Test the factor framework."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("TESTING FACTOR FRAMEWORK")
    print("="*60)
    
    # Test 1: Registry functionality
    print("\n1. Testing Registry")
    print("-" * 40)
    
    # List registered factors
    factors = FactorRegistry.list_factors()
    print(f"Registered factors: {factors}")
    
    # Get factor by name
    factor_class = FactorRegistry.get("test_factor")
    print(f"Retrieved factor class: {factor_class}")
    
    # Create factor instance
    factor = FactorRegistry.create("test_factor")
    print(f"Created factor instance: {factor}")
    
    # Test 2: Factor calculation
    print("\n2. Testing Factor Calculation")
    print("-" * 40)
    
    # Create test data
    test_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
        'date': pd.date_range('2024-01-01', periods=3).tolist() * 2,
        'close': [150.0, 151.0, 152.0, 300.0, 301.0, 302.0]
    })
    
    print("Test data:")
    print(test_data)
    
    # Calculate factor
    try:
        result = factor.calculate(test_data)
        print(f"\nFactor values:\n{result}")
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
    
    # Test 3: Factor with diagnostics
    print("\n3. Testing Factor with Diagnostics")
    print("-" * 40)
    
    # Add some edge cases to test data
    test_data_extended = test_data.copy()
    test_data_extended = pd.concat([
        test_data_extended,
        pd.DataFrame({
            'symbol': ['NVDA', 'AMZN'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'close': [np.nan, 0.0]  # Test null and zero values
        })
    ])
    
    values, diagnostics = factor.calculate_with_diagnostics(test_data_extended)
    
    print(f"Factor values:\n{values}")
    print(f"\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    
    # Test 4: Missing columns validation
    print("\n4. Testing Data Validation")
    print("-" * 40)
    
    bad_data = pd.DataFrame({
        'symbol': ['AAPL'],
        'price': [150.0]  # Wrong column name
    })
    
    try:
        factor.calculate(bad_data)
    except ValueError as e:
        print(f"✓ Validation correctly caught error: {e}")
    
    # Test 5: Registry error handling
    print("\n5. Testing Registry Error Handling")
    print("-" * 40)
    
    try:
        FactorRegistry.get("nonexistent_factor")
    except ValueError as e:
        print(f"✓ Registry correctly caught error: {e}")
    
    print("\n✓ Factor framework tests completed!")
    

if __name__ == "__main__":
    main()