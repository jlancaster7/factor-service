"""
Snowflake connector with optimized bulk insert using pandas
"""

import time
import json
import pandas as pd
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Union
from datetime import date, datetime

import snowflake.connector
from snowflake.connector import DictCursor
from snowflake.connector.pandas_tools import write_pandas
from loguru import logger

from ..utils.config import SnowflakeConfig


class SnowflakeConnector:
    """Manages Snowflake database connections and operations"""

    def __init__(self, config: SnowflakeConfig, use_pooling: bool = False):
        """
        Initialize Snowflake connector

        Args:
            config: Snowflake configuration
            use_pooling: Whether to use connection pooling (reuse connections)
        """
        self.config = config
        self.use_pooling = use_pooling
        self._connection = None

    def connect(self) -> None:
        """Establish connection to Snowflake"""
        if self._connection and not self._connection.is_closed():
            logger.debug("Reusing existing Snowflake connection")
            return

        logger.info("Connecting to Snowflake...")
        self._connection = snowflake.connector.connect(
            account=self.config.account,
            user=self.config.user,
            password=self.config.password,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema,
            role=self.config.role,
            insecure_mode=True,
        )
        logger.info("Successfully connected to Snowflake")

    def disconnect(self) -> None:
        """Close Snowflake connection"""
        if self.use_pooling:
            # Keep connection alive for reuse
            logger.debug("Keeping connection alive for reuse")
        else:
            if self._connection and not self._connection.is_closed():
                self._connection.close()
                logger.info("Disconnected from Snowflake")

    @contextmanager
    def cursor(self, dict_cursor: bool = True):
        """Context manager for cursor operations"""
        cursor_class = DictCursor if dict_cursor else None
        cursor = self._connection.cursor(cursor_class)
        try:
            yield cursor
        finally:
            cursor.close()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def execute(self, query: str, params: Optional[Union[tuple, dict]] = None) -> None:
        """
        Execute a query without returning results

        Args:
            query: SQL query to execute
            params: Query parameters
        """
        with self.cursor() as cursor:
            logger.debug(f"Executing query: {query}")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            logger.debug("Query executed successfully")

    def execute_with_rowcount(
        self, query: str, params: Optional[Union[tuple, dict]] = None
    ) -> int:
        """
        Execute a query and return the number of affected rows

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Number of rows affected
        """
        with self.cursor() as cursor:
            logger.debug(f"Executing query: {query}")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            rowcount = cursor.rowcount if cursor.rowcount is not None else 0
            logger.debug(f"Query executed successfully, affected {rowcount} rows")
            return rowcount

    def fetch_all(
        self, query: str, params: Optional[Union[tuple, dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and fetch all results

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of dictionaries containing the results
        """
        with self.cursor() as cursor:
            logger.debug(f"Fetching data with query: {query}")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            logger.debug(f"Fetched {len(results)} rows")
            return results

    def fetch_one(
        self, query: str, params: Optional[Union[tuple, dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a query and fetch one result

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Dictionary containing the result or None
        """
        with self.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()

    def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """
        Bulk insert data into a table using pandas write_pandas for optimal performance

        Args:
            table: Target table name (can include schema)
            data: List of dictionaries containing the data

        Returns:
            Number of rows inserted
        """
        if not data:
            logger.warning("No data to insert")
            return 0

        logger.info(f"Bulk inserting {len(data)} rows into {table}")

        # Common VARIANT column names in our schema
        variant_columns = {"raw_data", "metadata", "error_details"}

        # Check if any columns need special handling for VARIANT
        columns = list(data[0].keys())
        has_variant = any(col.lower() in variant_columns for col in columns)

        # Ensure columns are uppercase for Snowflake (but keep track of original for data access)
        original_columns = columns.copy()
        columns = [col.upper() for col in columns]

        # Convert data to DataFrame
        if has_variant:
            # Pre-process VARIANT columns to JSON strings
            processed_data = []
            for record in data:
                processed_record = {}
                for col, value in record.items():
                    if col.lower() in variant_columns and not isinstance(value, str):
                        processed_record[col] = json.dumps(value)
                    else:
                        processed_record[col] = value
                processed_data.append(processed_record)
            df = pd.DataFrame(processed_data)
        else:
            # Direct conversion for non-VARIANT data
            df = pd.DataFrame(data)

        # Ensure column names are uppercase for Snowflake
        df.columns = [col.upper() for col in df.columns]

        try:
            # Parse table name and schema
            parts = table.split(".")
            if len(parts) == 2:
                schema_name = parts[0]
                table_name = parts[1]
            else:
                schema_name = self._connection.schema
                table_name = table

            # Use write_pandas for fast bulk insert
            success, nchunks, nrows, _ = write_pandas(
                conn=self._connection,
                df=df,
                table_name=table_name,
                database=self._connection.database,
                schema=schema_name,
                auto_create_table=False,
                overwrite=False,
                use_logical_type=True,
            )

            if success:
                logger.info(
                    f"Bulk insert completed for {table} - inserted {nrows} rows in {nchunks} chunks"
                )
                return nrows
            else:
                raise Exception("write_pandas failed")

        except Exception as e:
            logger.error(f"Bulk insert with write_pandas failed: {e}")
            # Fall back to executemany method
            logger.warning("Falling back to executemany method")
            return self._bulk_insert_fallback(
                table, data, original_columns, has_variant
            )

    def _bulk_insert_fallback(
        self,
        table: str,
        data: List[Dict[str, Any]],
        columns: List[str],
        has_variant: bool,
    ) -> int:
        """Fallback method for bulk insert if write_pandas fails"""
        variant_columns = {"raw_data", "metadata", "error_details"}

        # Ensure uppercase columns for SQL query
        upper_columns = [col.upper() for col in columns]

        with self.cursor() as cursor:
            if has_variant:
                # Single-row inserts for VARIANT columns
                inserted = 0
                for record in data:
                    try:
                        row_values = []
                        placeholders = []
                        for col in columns:
                            value = record[col]
                            if col.lower() in variant_columns:
                                if not isinstance(value, str):
                                    value = json.dumps(value)
                                placeholders.append(f"PARSE_JSON(%s)")
                            else:
                                placeholders.append("%s")
                            row_values.append(value)

                        query = f"INSERT INTO {table} ({','.join(upper_columns)}) SELECT {','.join(placeholders)}"
                        cursor.execute(query, tuple(row_values))
                        inserted += 1

                        if inserted % 100 == 0:
                            logger.debug(f"Inserted {inserted} rows...")
                    except Exception as e:
                        logger.error(f"Failed to insert row: {e}")
                        raise

                logger.info(
                    f"Fallback bulk insert completed for {table} - inserted {inserted} rows"
                )
                return inserted
            else:
                # Use executemany for non-VARIANT data
                values = []
                for record in data:
                    row_values = [record[col] for col in columns]
                    values.append(tuple(row_values))

                query = f"INSERT INTO {table} ({','.join(upper_columns)}) VALUES ({','.join(['%s'] * len(columns))})"
                cursor.executemany(query, values)
                inserted = cursor.rowcount
                logger.info(
                    f"Fallback bulk insert completed for {table} - inserted {inserted} rows"
                )
                return inserted

    def merge(
        self,
        table: str,
        data: List[Dict[str, Any]],
        merge_keys: List[str],
        update_columns: Optional[List[str]] = None,
    ) -> int:
        """
        Merge data into table using MERGE statement

        Args:
            table: Target table name
            data: List of dictionaries containing the data
            merge_keys: List of column names to use for matching
            update_columns: List of columns to update (if None, updates all non-key columns)

        Returns:
            Number of rows affected
        """
        if not data:
            logger.warning("No data to merge")
            return 0

        # Get all columns from first record
        columns = list(data[0].keys())

        # If update_columns not specified, use all non-key columns
        if update_columns is None:
            update_columns = [col for col in columns if col not in merge_keys]

        # Create temporary table with same structure
        temp_table = f"{table}_TEMP_{int(time.time() * 1000)}"

        with self.cursor() as cursor:
            # Create temp table
            cursor.execute(f"CREATE TEMPORARY TABLE {temp_table} LIKE {table}")

            # Insert data into temp table using bulk_insert
            self.bulk_insert(temp_table, data)

            # Build MERGE statement
            merge_conditions = " AND ".join(
                [f"target.{key} = source.{key}" for key in merge_keys]
            )
            update_sets = ", ".join([f"{col} = source.{col}" for col in update_columns])

            # Build insert columns and values
            insert_columns = ", ".join(columns)
            insert_values = ", ".join([f"source.{col}" for col in columns])

            merge_query = f"""
            MERGE INTO {table} AS target
            USING {temp_table} AS source
            ON {merge_conditions}
            WHEN MATCHED THEN 
                UPDATE SET {update_sets}
            WHEN NOT MATCHED THEN 
                INSERT ({insert_columns})
                VALUES ({insert_values})
            """

            logger.debug(f"Executing MERGE: {merge_query}")
            cursor.execute(merge_query)
            rows_affected = cursor.rowcount

            # Drop temp table
            cursor.execute(f"DROP TABLE {temp_table}")

            logger.info(f"Merge completed for {table} - affected {rows_affected} rows")
            return rows_affected
