# Equity Factors Service

A Python service for calculating equity factors for quantitative analysis.

## Overview

This service provides a framework for calculating various equity factors including:
- Momentum factors
- Value factors  
- Technical indicators
- Quality factors

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

Calculate factors for test companies:
```bash
python scripts/calculate_factors.py
```

## Project Structure

```
equity-factors-service/
├── src/
│   ├── factors/       # Factor implementations
│   ├── data/          # Data loading utilities
│   └── config.py      # Configuration management
├── tests/             # Unit tests
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Standalone scripts
└── sql/              # Database schema definitions
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src tests
```

Type check:
```bash
mypy src
```