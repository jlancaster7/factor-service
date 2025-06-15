# Phase 1 Implementation Plan: Foundation (Weeks 1-3)

## Overview
This document provides a detailed, methodical plan for implementing the foundation of the equity factors service. Each story is broken down with clear objectives, design decisions, and implementation steps.

**Last Updated**: 2025-06-14  
**Current Status**: Environment Setup Complete, Ready for Factor Framework

## Progress Summary

| Story | Description | Status | Notes |
|-------|-------------|--------|-------|
| 0 | Database Schema Setup | ✅ Complete | All tables created, initial factors registered |
| 1 | Project Structure | ✅ Complete | Clean structure, dependencies installed |
| 2 | Base Factor Framework | 🚧 Next Task | Ready to implement |
| 3 | First Three Factors | ⏳ Pending | momentum_12_1, book_to_market, rsi_14 |
| 4 | Data Loading | ✅ Complete | Using production connector with DataLoader |
| 5 | Testing Framework | ⏳ Pending | Ready once factors implemented |
| 6 | Runner Script | ⏳ Pending | Will test with 50 companies |

## Key Achievements
- **Integrated production Snowflake connector** from data service (major win!)
- **Verified data availability**: 50 companies, 5 years of data, no quality issues
- **Created DataLoader** for seamless List[Dict] to DataFrame conversion
- **All scripts tested** and working with new connector

## Story 0: Database Schema Setup ✅ COMPLETE

### Objective
Create the required Snowflake tables for factor storage before implementing the framework.

### Status
✅ **Completed on 2025-06-14**
- All tables created successfully
- Initial factors registered in DIM_FACTOR
- Verified with setup_database.py script

### Implementation Steps

#### 0.1 Create Factor Schema Tables
```sql
-- sql/01_create_factor_tables.sql

-- Staging layer for calculated factors
CREATE TABLE IF NOT EXISTS STAGING.STG_FACTOR_VALUES (
    symbol VARCHAR(10),
    date DATE,
    factor_name VARCHAR(100),
    factor_value FLOAT,
    calculation_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (symbol, date, factor_name)
);

-- Analytics layer - Factor dimension
CREATE TABLE IF NOT EXISTS ANALYTICS.DIM_FACTOR (
    factor_key INTEGER IDENTITY(1,1),
    factor_name VARCHAR(100) UNIQUE NOT NULL,
    factor_category VARCHAR(50),
    factor_subcategory VARCHAR(50),
    calculation_frequency VARCHAR(20) DEFAULT 'daily',
    lookback_days INTEGER,
    description TEXT,
    formula TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    updated_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (factor_key)
);

-- Analytics layer - Daily factor values
CREATE TABLE IF NOT EXISTS ANALYTICS.FACT_DAILY_FACTORS (
    factor_value_key INTEGER IDENTITY(1,1),
    company_key INTEGER,
    date_key INTEGER,
    factor_key INTEGER,
    raw_value FLOAT,
    z_score FLOAT,
    percentile_rank FLOAT,
    sector_z_score FLOAT,
    sector_percentile_rank FLOAT,
    calculation_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (factor_value_key),
    FOREIGN KEY (company_key) REFERENCES DIM_COMPANY(company_key),
    FOREIGN KEY (date_key) REFERENCES DIM_DATE(date_key),
    FOREIGN KEY (factor_key) REFERENCES DIM_FACTOR(factor_key),
    UNIQUE (company_key, date_key, factor_key)
);

-- Calculation log for monitoring
CREATE TABLE IF NOT EXISTS ANALYTICS.FACTOR_CALCULATION_LOG (
    log_id INTEGER IDENTITY(1,1),
    calculation_date DATE,
    factor_name VARCHAR(100),
    symbols_processed INTEGER,
    records_calculated INTEGER,
    null_count INTEGER,
    inf_count INTEGER,
    calculation_time_seconds FLOAT,
    status VARCHAR(20),
    error_message TEXT,
    created_timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (log_id)
);

-- Insert our initial factors into DIM_FACTOR
INSERT INTO ANALYTICS.DIM_FACTOR (factor_name, factor_category, lookback_days, description, formula)
VALUES 
    ('momentum_12_1', 'momentum', 252, '12-month minus 1-month price momentum', '(Price_t-21 / Price_t-252) - 1'),
    ('book_to_market', 'value', 0, 'Book value to market value ratio', 'Total Equity / (Price * Shares Outstanding)'),
    ('rsi_14', 'technical', 14, '14-day Relative Strength Index', 'RSI = 100 - (100 / (1 + RS))');
```

#### 0.2 Create Schema Setup Script
```python
# scripts/setup_database.py
#!/usr/bin/env python
"""
Set up required database tables for factor framework
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = Config()
    config.validate()
    
    connector = SnowflakeConnector(config)
    
    # Read SQL file
    sql_file = Path('sql/01_create_factor_tables.sql')
    with open(sql_file, 'r') as f:
        sql_statements = f.read().split(';')
    
    # Execute each statement
    for statement in sql_statements:
        statement = statement.strip()
        if statement:
            logger.info(f"Executing: {statement[:50]}...")
            try:
                connector.execute(statement)
                logger.info("Success")
            except Exception as e:
                logger.error(f"Failed: {e}")
    
    connector.close()
    logger.info("Database setup complete")

if __name__ == "__main__":
    main()
```

#### 0.3 Data Availability Check Script
```python
# scripts/check_data_availability.py
#!/usr/bin/env python
"""
Quick script to check what data is available in Snowflake
"""
import sys
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector

def main():
    config = Config()
    connector = SnowflakeConnector(config)
    
    # Check price data
    print("=== PRICE DATA CHECK ===")
    query = """
    SELECT 
        c.symbol,
        COUNT(*) as record_count,
        MIN(d.date) as min_date,
        MAX(d.date) as max_date,
        COUNT(DISTINCT d.date) as days_count
    FROM FACT_DAILY_PRICES p
    JOIN DIM_COMPANY c ON p.company_key = c.company_key
    JOIN DIM_DATE d ON p.date_key = d.date_key
    WHERE c.symbol IN ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    price_summary = connector.query(query)
    print(price_summary)
    
    # Check fundamental data
    print("\n=== FUNDAMENTAL DATA CHECK ===")
    query = """
    SELECT 
        c.symbol,
        COUNT(*) as quarters_count,
        MIN(fiscal_date_key) as earliest_quarter,
        MAX(fiscal_date_key) as latest_quarter
    FROM FACT_FINANCIALS f
    JOIN DIM_COMPANY c ON f.company_key = c.company_key
    WHERE c.symbol IN ('AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL')
    GROUP BY c.symbol
    ORDER BY c.symbol
    """
    
    fundamental_summary = connector.query(query)
    print(fundamental_summary)
    
    connector.close()

if __name__ == "__main__":
    main()
```

## Story 1: Project Structure and Development Environment ✅ COMPLETE

### Objective
Set up a clean, simple project structure that supports future growth without over-engineering.

### Status
✅ **Completed on 2025-06-14**
- Created clean directory structure
- Set up virtual environment
- Installed all dependencies
- Created .gitignore and README.md

### Design Decisions
1. **Python 3.9+** - Modern Python with type hints support
2. **Simple flat structure initially** - Avoid deep nesting until needed
3. **Minimal dependencies** - Start with core libraries only
4. **Configuration via environment variables** - Simple and cloud-friendly

### Implementation Steps

#### 1.1 Create Basic Directory Structure
```
equity-factors-service/
├── src/
│   ├── __init__.py
│   ├── factors/          # Factor implementations
│   ├── data/             # Data loading utilities
│   └── config.py         # Configuration management
├── tests/
│   ├── __init__.py
│   └── conftest.py       # Pytest configuration
├── scripts/              # Standalone scripts
├── notebooks/            # Jupyter notebooks for exploration
├── requirements.txt      # Core dependencies
├── requirements-dev.txt  # Development dependencies
├── setup.py             # Package setup
├── .env.example         # Environment variables template
├── .gitignore
└── README.md
```

#### 1.2 Set Up Dependencies
**requirements.txt** (keeping it minimal):
```
pandas>=1.5.0,<2.0.0
numpy>=1.23.0,<2.0.0
python-dotenv>=0.20.0
snowflake-connector-python>=3.0.0
```

**requirements-dev.txt**:
```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
mypy>=1.0.0
jupyter>=1.0.0
```

#### 1.3 Simple Configuration Management
```python
# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """Simple configuration using environment variables"""
    # Snowflake
    snowflake_account: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    snowflake_user: str = os.getenv("SNOWFLAKE_USER", "")
    snowflake_password: str = os.getenv("SNOWFLAKE_PASSWORD", "")
    snowflake_warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE", "")
    snowflake_database: str = os.getenv("SNOWFLAKE_DATABASE", "")
    snowflake_schema: str = os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS")
    
    # Application
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def validate(self):
        """Ensure required config is present"""
        required = ["snowflake_account", "snowflake_user", "snowflake_password"]
        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(f"Missing required config: {missing}")
```

## Story 2: Base Factor Framework 🚧 IN PROGRESS

### Objective
Create a simple, extensible base class for all factors that enforces consistent behavior.

### Status
🚧 **In Progress**
- This is our next task
- Will implement BaseFactor class and FactorRegistry

### Design Decisions
1. **Abstract base class** - Enforce interface compliance
2. **Minimal required methods** - Don't over-specify
3. **Pandas-centric** - Use DataFrame/Series for familiarity
4. **No complex inheritance** - Keep hierarchy flat

### Implementation Steps

#### 2.1 Create Base Factor Class
```python
# src/factors/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BaseFactor(ABC):
    """Base class for all factor calculations"""
    
    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        logger.info(f"Initialized factor: {name}")
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate factor values for all securities in the DataFrame.
        
        Args:
            data: DataFrame with required columns for calculation
            
        Returns:
            Series indexed by symbol with factor values
        """
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of DataFrame columns required for calculation"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate that data contains required columns"""
        required = self.get_required_columns()
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def calculate_with_diagnostics(self, data: pd.DataFrame) -> tuple:
        """
        Calculate factor with diagnostic information
        
        Returns:
            (factor_values, diagnostics_dict)
        """
        # Calculate
        values = self.calculate(data)
        
        # Generate diagnostics
        diagnostics = {
            'total_count': len(values),
            'null_count': values.isnull().sum(),
            'inf_count': np.isinf(values).sum() if values.dtype in ['float64', 'float32'] else 0,
            'zero_count': (values == 0).sum(),
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'percentiles': {
                '25%': values.quantile(0.25),
                '50%': values.quantile(0.50),
                '75%': values.quantile(0.75)
            }
        }
        
        # Log warnings
        if diagnostics['null_count'] > 0:
            logger.warning(f"{self.name}: {diagnostics['null_count']} null values found")
        if diagnostics['inf_count'] > 0:
            logger.warning(f"{self.name}: {diagnostics['inf_count']} infinite values found")
            
        return values, diagnostics
```

#### 2.2 Create Factor Registry
```python
# src/factors/registry.py
from typing import Dict, Type
from .base import BaseFactor
import logging

logger = logging.getLogger(__name__)

class FactorRegistry:
    """Simple registry to track available factors"""
    
    _factors: Dict[str, Type[BaseFactor]] = {}
    
    @classmethod
    def register(cls, factor_class: Type[BaseFactor]) -> Type[BaseFactor]:
        """Register a factor class"""
        instance = factor_class()
        cls._factors[instance.name] = factor_class
        logger.info(f"Registered factor: {instance.name}")
        return factor_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseFactor]:
        """Get a factor class by name"""
        if name not in cls._factors:
            raise ValueError(f"Unknown factor: {name}")
        return cls._factors[name]
    
    @classmethod
    def list_factors(cls) -> List[str]:
        """List all registered factor names"""
        return list(cls._factors.keys())
```

## Story 3: First Three Factor Implementations ⏳ PENDING

### Objective
Implement three diverse factors to validate the framework design.

### Status
⏳ **Pending**
- Waiting for base framework completion

### Design Decisions
1. **Start simple** - Basic calculations first
2. **Handle edge cases explicitly** - NaN, inf, missing data
3. **Log important steps** - Aid debugging
4. **Minimal optimization** - Focus on correctness

### Implementation Steps

#### 3.1 Momentum Factor (Price-based)
```python
# src/factors/momentum.py
import pandas as pd
import numpy as np
from .base import BaseFactor
from .registry import FactorRegistry

@FactorRegistry.register
class Momentum12_1(BaseFactor):
    """12-month minus 1-month momentum factor"""
    
    def __init__(self):
        super().__init__(name="momentum_12_1", category="momentum")
        self.lookback_days = 252  # ~12 months
        self.skip_days = 21      # ~1 month
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate 12-1 momentum for each symbol"""
        self.validate_data(data)
        
        results = {}
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < self.lookback_days:
                results[symbol] = np.nan
                continue
            
            # Get prices
            current_price = symbol_data['adj_close'].iloc[-1]
            start_price = symbol_data['adj_close'].iloc[-(self.lookback_days)]
            skip_price = symbol_data['adj_close'].iloc[-(self.skip_days)]
            
            # Calculate momentum
            momentum = (skip_price / start_price) - 1
            results[symbol] = momentum
        
        return pd.Series(results, name=self.name)
    
    def get_required_columns(self) -> List[str]:
        return ['symbol', 'date', 'adj_close']
```

#### 3.2 Book-to-Market Factor (Fundamental)
```python
# src/factors/value.py
@FactorRegistry.register
class BookToMarket(BaseFactor):
    """Book value to market value ratio"""
    
    def __init__(self):
        super().__init__(name="book_to_market", category="value")
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate B/M ratio using latest available data"""
        self.validate_data(data)
        
        # Get latest data for each symbol
        latest = data.sort_values('date').groupby('symbol').last()
        
        # Calculate book value per share
        book_value_per_share = latest['total_equity'] / latest['shares_outstanding']
        
        # Calculate B/M ratio
        book_to_market = book_value_per_share / latest['close_price']
        
        # Handle edge cases
        book_to_market = book_to_market.replace([np.inf, -np.inf], np.nan)
        
        return book_to_market.rename(self.name)
    
    def get_required_columns(self) -> List[str]:
        return ['symbol', 'date', 'total_equity', 'shares_outstanding', 'close_price']
```

#### 3.3 RSI Factor (Technical)
```python
# src/factors/technical.py
@FactorRegistry.register
class RSI14(BaseFactor):
    """14-day Relative Strength Index"""
    
    def __init__(self):
        super().__init__(name="rsi_14", category="technical")
        self.period = 14
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate 14-day RSI for each symbol"""
        self.validate_data(data)
        
        results = {}
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol].sort_values('date')
            
            if len(symbol_data) < self.period + 1:
                results[symbol] = np.nan
                continue
            
            # Calculate price changes
            prices = symbol_data['adj_close'].values
            changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[-self.period:])
            avg_loss = np.mean(losses[-self.period:])
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            results[symbol] = rsi
        
        return pd.Series(results, name=self.name)
    
    def get_required_columns(self) -> List[str]:
        return ['symbol', 'date', 'adj_close']
```

## Story 4: Basic Data Loading ✅ COMPLETE

### Objective
Create simple utilities to load data from Snowflake for factor calculation.

### Status
✅ **Completed on 2025-06-14**
- Integrated production Snowflake connector from data service
- Created DataLoader helper class for DataFrame conversion
- Tested with real data - all working

### Design Decisions
1. **Connection pooling later** - Start with simple connections
2. **Pandas read_sql** - Familiar and straightforward
3. **Explicit date handling** - Avoid timezone issues
4. **Query templates** - Reusable SQL patterns

### Implementation Steps

#### 4.1 Snowflake Connection Manager
```python
# src/data/snowflake_connector.py
import snowflake.connector
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class SnowflakeConnector:
    """Simple Snowflake connection manager"""
    
    def __init__(self, config):
        self.config = config
        self._connection = None
    
    def connect(self):
        """Create connection to Snowflake"""
        if self._connection is None:
            self._connection = snowflake.connector.connect(
                account=self.config.snowflake_account,
                user=self.config.snowflake_user,
                password=self.config.snowflake_password,
                warehouse=self.config.snowflake_warehouse,
                database=self.config.snowflake_database,
                schema=self.config.snowflake_schema
            )
            logger.info("Connected to Snowflake")
    
    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        self.connect()
        return pd.read_sql(sql, self._connection, params=params)
    
    def close(self):
        """Close connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
```

#### 4.2 Factor Data Loader
```python
# src/data/factor_data_loader.py
class FactorDataLoader:
    """Load data required for factor calculations"""
    
    def __init__(self, connector: SnowflakeConnector):
        self.connector = connector
    
    def load_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Load daily price data"""
        query = """
        SELECT 
            c.symbol,
            d.date,
            p.adj_close,
            p.close_price,
            p.volume
        FROM FACT_DAILY_PRICES p
        JOIN DIM_COMPANY c ON p.company_key = c.company_key
        JOIN DIM_DATE d ON p.date_key = d.date_key
        WHERE c.symbol IN ({placeholders})
          AND d.date BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY c.symbol, d.date
        """
        
        placeholders = ','.join(['%s'] * len(symbols))
        query = query.format(placeholders=placeholders)
        
        params = {'start_date': start_date, 'end_date': end_date}
        return self.connector.query(query, params)
    
    def load_fundamental_data(self, symbols: List[str], as_of_date: str) -> pd.DataFrame:
        """Load point-in-time fundamental data"""
        query = """
        WITH latest_financials AS (
            SELECT 
                f.*,
                c.symbol,
                ROW_NUMBER() OVER (PARTITION BY f.company_key ORDER BY f.accepted_date DESC) as rn
            FROM FACT_FINANCIALS f
            JOIN DIM_COMPANY c ON f.company_key = c.company_key
            WHERE c.symbol IN ({placeholders})
              AND f.accepted_date <= %(as_of_date)s
        )
        SELECT * FROM latest_financials WHERE rn = 1
        """
        
        placeholders = ','.join(['%s'] * len(symbols))
        query = query.format(placeholders=placeholders)
        
        return self.connector.query(query, {'as_of_date': as_of_date})
```

## Story 5: Testing Framework and Data Validation ⏳ PENDING

### Objective
Set up testing that uses both unit tests AND real data validation.

### Status
⏳ **Pending**
- Framework ready, waiting for factor implementations

### Design Decisions
1. **Pytest** for unit tests
2. **Jupyter notebooks** for visual data inspection
3. **Test with real data** - Not just synthetic
4. **Validate results make sense** - Check distributions, outliers

### Implementation Steps

#### 5.1 Test Fixtures
```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing"""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    symbols = ['AAPL', 'MSFT']
    
    data = []
    for symbol in symbols:
        for i, date in enumerate(dates):
            # Create realistic price movement
            base_price = 100 if symbol == 'AAPL' else 200
            price = base_price * (1 + 0.001 * i + 0.01 * np.random.randn())
            
            data.append({
                'symbol': symbol,
                'date': date,
                'adj_close': price,
                'close_price': price,
                'volume': 1000000
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_fundamental_data():
    """Create sample fundamental data for testing"""
    return pd.DataFrame([
        {
            'symbol': 'AAPL',
            'date': '2023-12-31',
            'total_equity': 50000000000,
            'shares_outstanding': 15500000000,
            'close_price': 195.0
        },
        {
            'symbol': 'MSFT', 
            'date': '2023-12-31',
            'total_equity': 100000000000,
            'shares_outstanding': 7500000000,
            'close_price': 375.0
        }
    ])
```

#### 5.2 Factor Tests
```python
# tests/test_factors.py
def test_momentum_factor(sample_price_data):
    """Test momentum factor calculation"""
    from src.factors.momentum import Momentum12_1
    
    factor = Momentum12_1()
    results = factor.calculate(sample_price_data)
    
    # Check results
    assert len(results) == 2  # Two symbols
    assert all(isinstance(v, (float, np.floating)) or pd.isna(v) for v in results.values)
    assert results.name == 'momentum_12_1'

def test_book_to_market_factor(sample_fundamental_data):
    """Test book-to-market factor calculation"""
    from src.factors.value import BookToMarket
    
    factor = BookToMarket()
    results = factor.calculate(sample_fundamental_data)
    
    # Check specific calculation
    aapl_bm = results['AAPL']
    expected = (50000000000 / 15500000000) / 195.0
    assert abs(aapl_bm - expected) < 0.0001
```

#### 5.3 Data Validation Notebooks
```python
# notebooks/01_test_data_loading.ipynb
"""
Test data loading and inspect actual values
"""
import sys
sys.path.append('..')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.factor_data_loader import FactorDataLoader

# Initialize
config = Config()
connector = SnowflakeConnector(config)
loader = FactorDataLoader(connector)

# Load sample data
symbols = ['AAPL', 'MSFT']
price_data = loader.load_price_data(symbols, '2024-01-01', '2024-01-31')

# Inspect the data
print(f"Shape: {price_data.shape}")
print(f"Columns: {price_data.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(price_data.head())

print(f"\nNull counts:")
print(price_data.isnull().sum())

print(f"\nSummary statistics:")
print(price_data.describe())

# Plot to visualize
import matplotlib.pyplot as plt

for symbol in symbols:
    symbol_data = price_data[price_data['symbol'] == symbol]
    plt.plot(symbol_data['date'], symbol_data['adj_close'], label=symbol)

plt.legend()
plt.title('Adjusted Close Prices')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# notebooks/02_validate_factors.ipynb
"""
Calculate factors and inspect results
"""
# ... imports ...

# Load data
symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
price_data = loader.load_price_data(symbols, '2023-01-01', '2024-01-01')

# Calculate momentum factor
from src.factors.momentum import Momentum12_1
factor = Momentum12_1()

values, diagnostics = factor.calculate_with_diagnostics(price_data)

print("Momentum Factor Results:")
print(f"Values:\n{values}")
print(f"\nDiagnostics:\n{diagnostics}")

# Visualize distribution
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
values.plot(kind='bar')
plt.title('Factor Values by Symbol')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
values.hist(bins=20)
plt.title('Factor Distribution')

plt.subplot(1, 3, 3)
sns.boxplot(y=values.values)
plt.title('Factor Box Plot')

plt.tight_layout()
plt.show()

# Check for reasonableness
print("\nReasonableness checks:")
print(f"All values between -1 and 2? {values.between(-1, 2).all()}")
print(f"Any extreme outliers? {(values.abs() > 3).any()}")
```

## Story 6: Simple Runner Script ⏳ PENDING

### Objective
Create a script to test factor calculations with our test companies.

### Status
⏳ **Pending**
- Note: We now have 50 companies available, not just 5

### Design Decisions
1. **Command-line interface** - Simple to use
2. **CSV output initially** - Easy to inspect
3. **Logging to console** - Immediate feedback
4. **No scheduling yet** - Manual execution

### Implementation Steps

```python
# scripts/calculate_factors.py
#!/usr/bin/env python
"""
Simple script to calculate factors for test companies
"""
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, '.')

from src.config import Config
from src.data.snowflake_connector import SnowflakeConnector
from src.data.factor_data_loader import FactorDataLoader
from src.factors.registry import FactorRegistry

# Import factors to register them
import src.factors.momentum
import src.factors.value
import src.factors.technical

def validate_results(results_df: pd.DataFrame):
    """Validate factor calculation results"""
    print("\n=== DATA QUALITY REPORT ===")
    
    for col in results_df.columns:
        print(f"\n{col}:")
        print(f"  Non-null: {results_df[col].notna().sum()}/{len(results_df)}")
        print(f"  Range: [{results_df[col].min():.4f}, {results_df[col].max():.4f}]")
        print(f"  Mean: {results_df[col].mean():.4f}")
        print(f"  Std: {results_df[col].std():.4f}")
        
        # Flag potential issues
        if results_df[col].isnull().all():
            print("  ⚠️  WARNING: All values are null!")
        elif results_df[col].nunique() == 1:
            print("  ⚠️  WARNING: All values are identical!")
        elif results_df[col].std() == 0:
            print("  ⚠️  WARNING: No variation in values!")

def main():
    # Setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = Config()
    config.validate()
    
    # Test companies
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL']
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    # Load data
    connector = SnowflakeConnector(config)
    loader = FactorDataLoader(connector)
    
    logging.info(f"Loading data for {symbols} from {start_date} to {end_date}")
    price_data = loader.load_price_data(symbols, str(start_date), str(end_date))
    fundamental_data = loader.load_fundamental_data(symbols, str(end_date))
    
    # Calculate factors
    results = pd.DataFrame(index=symbols)
    diagnostics = {}
    
    for factor_name in FactorRegistry.list_factors():
        logging.info(f"Calculating {factor_name}")
        factor_class = FactorRegistry.get(factor_name)
        factor = factor_class()
        
        # Use appropriate data
        if factor.category in ['momentum', 'technical']:
            data = price_data
        else:
            data = fundamental_data
        
        try:
            factor_values, factor_diagnostics = factor.calculate_with_diagnostics(data)
            results[factor_name] = factor_values
            diagnostics[factor_name] = factor_diagnostics
        except Exception as e:
            logging.error(f"Failed to calculate {factor_name}: {e}")
    
    # Validate results
    validate_results(results)
    
    # Save results
    output_file = f"factor_values_{end_date}.csv"
    results.to_csv(output_file)
    logging.info(f"Saved results to {output_file}")
    
    # Display summary
    print("\nFactor Values Summary:")
    print(results)
    
    # Display diagnostics
    print("\nFactor Diagnostics:")
    for factor_name, diag in diagnostics.items():
        print(f"\n{factor_name}:")
        print(f"  Nulls: {diag['null_count']}, Infs: {diag['inf_count']}")
        print(f"  Mean: {diag['mean']:.4f}, Std: {diag['std']:.4f}")
    
    connector.close()

if __name__ == "__main__":
    main()
```

## Development Timeline

### Week 1: Foundation
- Day 1: Create database schema and verify data availability
- Day 2: Set up project structure, dependencies, configuration
- Day 3-4: Implement base factor class and registry
- Day 5: Initial testing framework and data validation notebooks

### Week 2: Factor Implementation  
- Day 1-2: Implement and validate momentum factor with real data
- Day 3-4: Implement and validate book-to-market factor with real data
- Day 5: Implement and validate RSI factor with real data

### Week 3: Integration and Testing
- Day 1-2: Build data loader and Snowflake connector
- Day 3: Create runner script with data quality checks
- Day 4: Test end-to-end with real data, fix any issues
- Day 5: Documentation and cleanup

## Success Criteria
1. Three factors calculating correctly for 5 test companies with NO null values
2. Clean, simple, extensible codebase
3. Both unit tests AND real data validation passing
4. Manual execution producing reasonable, validated results
5. Clear documentation for adding new factors
6. Data quality checks showing expected distributions

## Next Steps
After Phase 1 completion:
1. Add more factors following established patterns
2. Implement parallel processing for performance
3. Add data quality checks and monitoring
4. Build incremental update logic