"""Setup configuration for equity factors service."""
from setuptools import setup, find_packages

setup(
    name="equity-factors-service",
    version="0.1.0",
    description="Equity factors calculation service",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0,<2.0.0",
        "numpy>=1.23.0,<2.0.0",
        "python-dotenv>=0.20.0",
        "snowflake-connector-python>=3.0.0",
    ],
)