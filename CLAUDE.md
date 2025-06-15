# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an equity factors service project. The codebase is currently being initialized.

## Core Development Principles

### Keep It Stupid Simple (KISS)
- Always choose the simplest solution that works
- Avoid over-engineering and premature optimization
- Clear, readable code is better than clever code
- Start with basic implementations and iterate

### Methodical Development Approach
- Plan and document each story before implementation
- Break down work into small, manageable phases
- Think through design decisions carefully
- Don't rush - thoughtful implementation is key
- Document the "why" behind decisions
- Make sure to not only create unit tests, but use actual data and view actual data as you are building components and make sure the results you're getting actually make sense. Sometimes just looking at the data you can see whats wrong immediately (like everything is null).

## Commands

### Development Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
```

### Database Setup
```bash
# Create factor framework tables
python scripts/setup_database.py

# Check data availability
python scripts/check_data_availability.py
```

### Testing
```bash
# Test configuration
python scripts/test_config.py

# Test Snowflake connection
python scripts/test_connection.py

# Run unit tests (once created)
pytest
```

### Linting & Type Checking
```bash
# Format code
black src tests scripts

# Type checking
mypy src

# Linting
flake8 src tests scripts
```

### Build & Run
```bash
# Calculate factors (once implemented)
python scripts/calculate_factors.py
```

## Architecture

### Data Layer
- **Snowflake Connector**: Production-ready connector from data service with bulk operations, connection pooling, and error handling
- **DataLoader**: Helper class that converts List[Dict] results to pandas DataFrames for factor calculations
- **Config Management**: Environment-based configuration with SnowflakeConfig compatibility

### Factor Framework (In Progress)
- **BaseFactor**: Abstract base class for all factor implementations
- **FactorRegistry**: Central registry for factor discovery and management
- **Factor Categories**: momentum, value, technical, quality

### Database Schema
- **STAGING.STG_FACTOR_VALUES**: Staging layer for calculated factors
- **ANALYTICS.DIM_FACTOR**: Factor metadata and definitions
- **ANALYTICS.FACT_DAILY_FACTORS**: Daily factor values with standardization
- **ANALYTICS.FACTOR_CALCULATION_LOG**: Monitoring and audit trail

## Key Considerations

As this is an equity factors service, future development should consider:
- Financial data accuracy and precision
- Performance optimization for large datasets
- Proper error handling for market data operations
- Security considerations for financial data