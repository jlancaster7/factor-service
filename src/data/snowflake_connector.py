"""Legacy import alias for local storage.

Snowflake has been replaced by DuckDB. New code should import
DatabaseConnector from src.data.database and pass src.config.Config.
No Snowflake SDK or credentials are used by this module.
"""
from .database import DatabaseConnector

SnowflakeConnector = DatabaseConnector
