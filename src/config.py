"""Simple configuration management using environment variables."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from .utils.config import SnowflakeConfig

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Simple configuration using environment variables."""
    
    # Application settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # For backward compatibility
    snowflake_account: str = os.getenv("SNOWFLAKE_ACCOUNT", "")
    snowflake_user: str = os.getenv("SNOWFLAKE_USER", "")
    snowflake_password: str = os.getenv("SNOWFLAKE_PASSWORD", "")
    snowflake_warehouse: str = os.getenv("SNOWFLAKE_WAREHOUSE", "")
    snowflake_database: str = os.getenv("SNOWFLAKE_DATABASE", "")
    snowflake_schema: str = os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS")
    
    def get_snowflake_config(self) -> SnowflakeConfig:
        """Get SnowflakeConfig instance for the data service connector."""
        return SnowflakeConfig.from_env()
    
    def validate(self):
        """Ensure required config is present."""
        # Delegate to SnowflakeConfig validation
        snowflake_config = self.get_snowflake_config()
        snowflake_config.validate()
    
    def __repr__(self):
        """Safe representation without sensitive data."""
        return (
            f"Config("
            f"snowflake_account={self.snowflake_account}, "
            f"snowflake_user={self.snowflake_user}, "
            f"snowflake_warehouse={self.snowflake_warehouse}, "
            f"snowflake_database={self.snowflake_database}, "
            f"snowflake_schema={self.snowflake_schema}, "
            f"log_level={self.log_level})"
        )