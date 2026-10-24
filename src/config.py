"""Local database and application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """Environment defaults are read when a configuration is created."""

    database_path: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_PATH", str(PROJECT_ROOT / "data" / "equity_factors.duckdb")
        )
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_SIZE", "1000"))
    )

    def validate(self):
        if not self.database_path.strip():
            raise ValueError("DATABASE_PATH must not be empty")
        if self.batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
