#!/usr/bin/env python3
"""Load CSV or Parquet source data into the local database."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config
from src.data.database import DatabaseConnector
from src.data.local_import import import_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["prices", "market-metrics"])
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    frame = (
        pd.read_parquet(args.file)
        if args.file.suffix.lower() == ".parquet"
        else pd.read_csv(args.file)
    )
    with DatabaseConnector(Config()) as db:
        count = import_frame(db, frame, args.kind)
    print(f"Imported {count:,} {args.kind} records")


if __name__ == "__main__":
    main()
