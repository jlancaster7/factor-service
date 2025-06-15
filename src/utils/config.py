"""Configuration classes compatible with the data service."""
from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class SnowflakeConfig:
    """Snowflake configuration matching the data service structure."""
    account: str
    user: str
    password: str
    warehouse: str
    database: str
    schema: str = "ANALYTICS"
    role: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'SnowflakeConfig':
        """Create config from environment variables."""
        return cls(
            account=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            user=os.getenv("SNOWFLAKE_USER", ""),
            password=os.getenv("SNOWFLAKE_PASSWORD", ""),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", ""),
            database=os.getenv("SNOWFLAKE_DATABASE", ""),
            schema=os.getenv("SNOWFLAKE_SCHEMA", "ANALYTICS"),
            role=os.getenv("SNOWFLAKE_ROLE")
        )
    
    def validate(self):
        """Ensure required config is present."""
        required = ["account", "user", "password"]
        missing = [f for f in required if not getattr(self, f)]
        if missing:
            raise ValueError(f"Missing required Snowflake config: {missing}")