# Equity Factors Service

A Python service for calculating equity factors for quantitative analysis.

## Overview

This service provides a framework for calculating various equity factors including:
- Momentum factors (momentum_12_1)
- Value factors (book_to_market)
- Technical indicators (rsi_14)
- Quality factors (coming soon)

### Current Status

✅ **Phase 1 Complete**: Base framework and initial three factors implemented and tested with real market data.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your Snowflake credentials
```

4. Set up database tables:
```bash
python scripts/setup_database.py
```

5. Verify data availability:
```bash
python scripts/check_data_availability.py
```

## Usage

Calculate all factors for test companies:
```bash
python scripts/calculate_all_factors.py
```

Test individual factors:
```bash
python scripts/test_momentum_factor.py
python scripts/test_book_to_market_simple.py  
python scripts/test_rsi_factor.py
```

## Project Structure

```
equity-factors-service/
├── src/
│   ├── factors/       # Factor implementations
│   │   ├── base.py         # BaseFactor abstract class
│   │   ├── registry.py     # Factor registry
│   │   ├── momentum.py     # Momentum factors
│   │   ├── value.py        # Value factors
│   │   └── technical.py    # Technical indicators
│   ├── data/          # Data loading utilities
│   │   ├── snowflake_connector.py  # Production Snowflake connector
│   │   └── data_loader.py          # DataFrame conversion utilities
│   ├── utils/         # Utility modules
│   │   └── config.py       # Snowflake config compatibility
│   └── config.py      # Configuration management
├── tests/             # Unit tests (to be implemented)
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Standalone scripts
├── sql/               # Database schema definitions
└── docs/              # Documentation
```

## Implemented Factors

### Momentum Factors
- **momentum_12_1**: 12-month minus 1-month price momentum
  - Lookback: 252 trading days
  - Skip: 21 trading days

### Value Factors  
- **book_to_market**: Book-to-market ratio (1/PB)
  - Uses PB_RATIO from FACT_MARKET_METRICS table
  - Point-in-time data to avoid look-ahead bias

### Technical Indicators
- **rsi_14**: 14-day Relative Strength Index
  - Uses Wilder's smoothing method
  - Range: 0-100 (>70 overbought, <30 oversold)

## Development

Run tests:
```bash
pytest  # Unit tests to be implemented
```

Format code:
```bash
black src tests scripts
```

Type check:
```bash
mypy src
```

Lint code:
```bash
flake8 src tests scripts
```

## Key Design Decisions

1. **KISS Principle**: Keep implementations simple and clear
2. **Production Snowflake Connector**: Reused from data service for consistency
3. **Automatic Type Conversion**: DataLoader handles Decimal-to-float conversion
4. **Extensible Framework**: Easy to add new factors via BaseFactor and registry