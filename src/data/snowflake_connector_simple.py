"""Simple Snowflake connection manager."""
import snowflake.connector
import pandas as pd
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SnowflakeConnector:
    """Simple Snowflake connection manager."""
    
    def __init__(self, config):
        """Initialize with configuration object."""
        self.config = config
        self._connection = None
    
    def connect(self):
        """Create connection to Snowflake."""
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
            logger.info(f"Database: {self.config.snowflake_database}")
            logger.info(f"Schema: {self.config.snowflake_schema}")
    
    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        self.connect()
        return pd.read_sql(sql, self._connection, params=params)
    
    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Execute SQL statement without returning results."""
        self.connect()
        cursor = self._connection.cursor()
        try:
            cursor.execute(sql, params)
            self._connection.commit()
        finally:
            cursor.close()
    
    def test_connection(self) -> bool:
        """Test the connection is working."""
        try:
            self.connect()
            cursor = self._connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to Snowflake version: {version}")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def close(self):
        """Close connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Closed Snowflake connection")