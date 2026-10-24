"""Persistent, local DuckDB storage for the factor pipeline."""

from pathlib import Path
import re
from uuid import uuid4

import duckdb
import pandas as pd

from src.config import Config, PROJECT_ROOT


def identifier(name):
    """Quote a possibly schema-qualified SQL identifier, never SQL text."""
    if not all(
        re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part) for part in name.split(".")
    ):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return ".".join('"' + part + '"' for part in name.split("."))


class DatabaseConnector:
    """One connection per instance; no network, credentials, or server required.

    Named Snowflake-style placeholders are bound as DuckDB parameters. Results
    use uppercase keys for compatibility with the existing ETL/data loader.
    """

    def __init__(self, config=None, use_pooling=False):
        self.config = config or Config()
        self._connection = None

    def connect(self):
        if self._connection is not None:
            return
        self.config.validate()
        location = self.config.database_path
        if location != ":memory:":
            path = Path(location).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            location = str(path)
        self._connection = duckdb.connect(location)

    def disconnect(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def initialize_schema(self):
        """Create source and output tables without altering existing data."""
        self.connect()
        self._connection.execute(Path(__file__).with_name("schema.sql").read_text())

    def _execute(self, query, params=None):
        self.connect()
        if isinstance(params, dict):
            query = re.sub(r"%\((\w+)\)s", r"$\1", query)
        elif params is not None:
            query = query.replace("%s", "?")
        return (
            self._connection.execute(query, params)
            if params is not None
            else self._connection.execute(query)
        )

    def execute(self, query, params=None):
        self._execute(query, params)

    def execute_with_rowcount(self, query, params=None):
        result = self._execute(query, params)
        row = result.fetchone()
        return int(row[0]) if row else 0

    def fetch_all(self, query, params=None):
        result = self._execute(query, params)
        columns = [col[0].upper() for col in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def fetch_one(self, query, params=None):
        rows = self.fetch_all(query, params)
        return rows[0] if rows else None

    def _insert(self, table, data, conflict=""):
        if not data:
            return 0
        self.connect()
        table_sql = identifier(table)
        columns = ", ".join(identifier(col) for col in data[0])
        frame = pd.DataFrame(data)
        relation = "_batch_" + uuid4().hex
        self._connection.register(relation, frame)
        try:
            self._connection.execute(
                f"INSERT INTO {table_sql} ({columns}) "
                f"SELECT {columns} FROM {relation} {conflict}"
            )
        finally:
            self._connection.unregister(relation)
        return len(data)

    def bulk_insert(self, table, data):
        return self._insert(table, data)

    def merge(self, table, data, merge_keys, update_columns=None):
        if not data:
            return 0
        # Reject ambiguous batches rather than silently choosing a duplicate.
        frame = pd.DataFrame(data)
        if not merge_keys or frame[merge_keys].isna().any().any():
            raise ValueError("Merge keys must be present and non-null")
        if frame.duplicated(merge_keys).any():
            raise ValueError("Duplicate merge keys in batch")
        if update_columns is None:
            update_columns = [col for col in data[0] if col not in merge_keys]
        keys = ", ".join(identifier(key) for key in merge_keys)
        if update_columns:
            updates = ", ".join(
                f"{identifier(col)} = excluded.{identifier(col)}"
                for col in update_columns
            )
            conflict = f"ON CONFLICT ({keys}) DO UPDATE SET {updates}"
        else:
            conflict = f"ON CONFLICT ({keys}) DO NOTHING"
        return self._insert(table, data, conflict)
