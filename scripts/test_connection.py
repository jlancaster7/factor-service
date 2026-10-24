#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.data.database import DatabaseConnector


def main():
    config = Config()
    with DatabaseConnector(config) as db:
        assert db.fetch_one("SELECT 1 AS ok")["OK"] == 1
        print(f"Local connection successful: {config.database_path}")
        for row in db.fetch_all(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'ANALYTICS' ORDER BY table_name"
        ):
            print("  " + row["TABLE_NAME"])


if __name__ == "__main__":
    main()
