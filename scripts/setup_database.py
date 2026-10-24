#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.database import DatabaseConnector


def main():
    config = Config()
    with DatabaseConnector(config) as db:
        db.initialize_schema()
        print(f"Local database ready: {config.database_path}")
        for row in db.fetch_all(
            "SELECT factor_name FROM ANALYTICS.DIM_FACTOR ORDER BY factor_name"
        ):
            print("  " + row["FACTOR_NAME"])


if __name__ == "__main__":
    main()
