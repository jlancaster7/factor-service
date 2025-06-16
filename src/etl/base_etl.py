"""
Base ETL framework for financial data pipelines
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import time
from loguru import logger

from src.data.snowflake_connector import SnowflakeConnector
from src.validation.factor_validator import FactorDataValidator as DataQualityValidator


class ETLStatus(Enum):
    """ETL job status enum"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ETLResult:
    """Result of an ETL job execution"""

    job_name: str
    status: ETLStatus
    start_time: datetime
    end_time: Optional[datetime]
    records_extracted: int
    records_transformed: int
    records_loaded: int
    errors: List[str]
    metadata: Dict[str, Any]

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "job_name": self.job_name,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "records_extracted": self.records_extracted,
            "records_transformed": self.records_transformed,
            "records_loaded": self.records_loaded,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class BaseETL(ABC):
    """Abstract base class for ETL pipelines"""

    def __init__(
        self,
        job_name: str,
        snowflake_connector: SnowflakeConnector,
        fmp_client: Optional[Any] = None,  # Made optional for factor ETL
        batch_size: int = 1000,
        max_retries: int = 3,
        retry_delay: int = 5,
        enable_monitoring: bool = True,
    ):
        """
        Initialize base ETL

        Args:
            job_name: Name of the ETL job
            snowflake_connector: Snowflake database connector
            fmp_client: FMP API client
            batch_size: Size of batches for processing
            max_retries: Maximum number of retries for failed operations
            retry_delay: Delay in seconds between retries
            enable_monitoring: Whether to persist job results to monitoring tables
        """
        self.job_name = job_name
        self.snowflake = snowflake_connector
        self.fmp_client = fmp_client  # Can be None for factor ETL
        self.transformer = None  # Not needed for factor ETL
        self.validator = DataQualityValidator()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_monitoring = enable_monitoring

        # Initialize ETL monitor if enabled
        self.monitor = None
        if self.enable_monitoring:
            try:
                from src.etl.etl_monitor import ETLMonitor
                self.monitor = ETLMonitor(snowflake_connector)
            except ImportError:
                logger.warning("ETLMonitor not available, disabling monitoring")
                self.enable_monitoring = False

        # Monitoring hooks
        self._pre_extract_hooks: List[Callable] = []
        self._post_extract_hooks: List[Callable] = []
        self._pre_transform_hooks: List[Callable] = []
        self._post_transform_hooks: List[Callable] = []
        self._pre_load_hooks: List[Callable] = []
        self._post_load_hooks: List[Callable] = []

        # Job result tracking
        self.result = ETLResult(
            job_name=job_name,
            status=ETLStatus.PENDING,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            records_extracted=0,
            records_transformed=0,
            records_loaded=0,
            errors=[],
            metadata={},
        )

    @abstractmethod
    def extract(self) -> List[Dict[str, Any]]:
        """
        Extract data from source

        Returns:
            List of raw data records
        """
        pass

    @abstractmethod
    def transform(
        self, raw_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Transform raw data

        Args:
            raw_data: List of raw data records

        Returns:
            Dict with 'raw' and 'staging' keys containing transformed records
        """
        pass

    @abstractmethod
    def load(self, transformed_data: Dict[str, List[Dict[str, Any]]]) -> int:
        """
        Load transformed data to destination

        Args:
            transformed_data: Dict with transformed records

        Returns:
            Number of records loaded
        """
        pass

    def run(self) -> ETLResult:
        """
        Execute the ETL pipeline

        Returns:
            ETL execution result
        """
        logger.info(f"Starting ETL job: {self.job_name}")
        self.result.status = ETLStatus.RUNNING
        self._current_job_id = None

        try:
            # Extract phase
            logger.info("Starting extraction phase")
            extract_start = time.time()
            self._run_hooks(self._pre_extract_hooks)
            raw_data = self._extract_with_retry()
            self.result.records_extracted = len(raw_data)
            self._run_hooks(self._post_extract_hooks, data=raw_data)
            extract_duration = time.time() - extract_start
            logger.info(f"Extracted {len(raw_data)} records in {extract_duration:.2f}s")
            self.result.metadata["extract_duration"] = extract_duration

            if not raw_data:
                logger.warning("No data extracted, ending job")
                self.result.status = ETLStatus.SUCCESS
                self.result.end_time = datetime.now(timezone.utc)
                return self.result

            # Transform phase
            logger.info("Starting transformation phase")
            transform_start = time.time()
            self._run_hooks(self._pre_transform_hooks, data=raw_data)
            transformed_data = self._transform_with_validation(raw_data)
            self.result.records_transformed = len(transformed_data.get("staging", []))
            self._run_hooks(self._post_transform_hooks, data=transformed_data)
            transform_duration = time.time() - transform_start
            logger.info(
                f"Transformed {self.result.records_transformed} records in {transform_duration:.2f}s"
            )
            self.result.metadata["transform_duration"] = transform_duration

            # Load phase
            logger.info("Starting load phase")
            load_start = time.time()
            self._run_hooks(self._pre_load_hooks, data=transformed_data)
            records_loaded = self._load_in_batches(transformed_data)
            self.result.records_loaded = records_loaded
            self._run_hooks(self._post_load_hooks, count=records_loaded)
            load_duration = time.time() - load_start
            logger.info(f"Loaded {records_loaded} records in {load_duration:.2f}s")
            self.result.metadata["load_duration"] = load_duration

            # Determine final status
            if self.result.errors:
                self.result.status = ETLStatus.PARTIAL
            else:
                self.result.status = ETLStatus.SUCCESS

        except Exception as e:
            logger.error(f"ETL job failed: {str(e)}")
            self.result.status = ETLStatus.FAILED
            self.result.errors.append(str(e))
            raise

        finally:
            self.result.end_time = datetime.now(timezone.utc)
            logger.info(f"ETL job completed: {self.result.status.value}")
            logger.info(f"Job duration: {self.result.duration_seconds:.2f} seconds")
            self._log_result_summary()

            # Save job result to monitoring tables if enabled
            if self.monitor:
                try:
                    self._current_job_id = self.monitor.save_job_result(self.result)
                    logger.info(f"Job result saved with ID: {self._current_job_id}")
                except Exception as e:
                    logger.error(f"Failed to save job result to monitoring: {e}")

        return self.result

    def _extract_with_retry(self) -> List[Dict[str, Any]]:
        """Extract data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self.extract()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Extract attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise
        return []  # This should never be reached, but satisfies mypy

    def _transform_with_validation(
        self, raw_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Transform data with validation"""
        transformed_data = self.transform(raw_data)

        # Validate transformed data
        for data_type, records in transformed_data.items():
            if data_type == "staging" and records:
                # Determine validation type based on record structure
                validation_type = self._infer_validation_type(records[0])
                if validation_type:
                    validation_result = self.validator.validate_batch(
                        records, validation_type
                    )

                    if validation_result["invalid_records"] > 0:
                        logger.warning(
                            f"Found {validation_result['invalid_records']} invalid records "
                            f"out of {validation_result['total_records']}"
                        )

                        # Save data quality issues if monitoring is enabled
                        if self.monitor and hasattr(self, "_current_job_id"):
                            quality_issues = []
                            for issue in validation_result["issues_by_record"]:
                                for issue_text in issue.get("issues", []):
                                    quality_issues.append(
                                        {
                                            "record_identifier": issue.get(
                                                "record_identifier"
                                            ),
                                            "issue_type": "VALIDATION_ERROR",
                                            "description": issue_text,
                                            "severity": "WARNING",
                                        }
                                    )

                            if quality_issues:
                                self.monitor.save_data_quality_issues(
                                    self._current_job_id,
                                    data_type.upper(),
                                    quality_issues,
                                )

                        for issue in validation_result["issues_by_record"]:
                            logger.debug(f"Validation issue: {issue}")

        return transformed_data

    def _load_in_batches(
        self, transformed_data: Dict[str, List[Dict[str, Any]]]
    ) -> int:
        """Load data in batches"""
        total_loaded = 0

        for data_layer, records in transformed_data.items():
            if not records:
                continue

            # Process in batches
            for i in range(0, len(records), self.batch_size):
                batch = records[i : i + self.batch_size]

                for attempt in range(self.max_retries):
                    try:
                        # Let the subclass handle the actual loading
                        loaded = self.load({data_layer: batch})
                        total_loaded += loaded
                        logger.debug(
                            f"Loaded batch {i // self.batch_size + 1} ({loaded} records)"
                        )
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(
                                f"Load attempt {attempt + 1} failed: {e}. Retrying..."
                            )
                            time.sleep(self.retry_delay)
                        else:
                            self.result.errors.append(f"Failed to load batch: {str(e)}")
                            raise

        return total_loaded

    def _infer_validation_type(self, record: Dict[str, Any]) -> Optional[str]:
        """Infer validation type from record structure"""
        if "company_name" in record:
            return "profile"
        elif "price_date" in record:
            return "price"
        elif "revenue" in record:
            return "income"
        elif "total_assets" in record:
            return "balance"
        elif "operating_cash_flow" in record:
            return "cashflow"
        return None

    def _run_hooks(self, hooks: List[Callable], **kwargs):
        """Run monitoring hooks"""
        for hook in hooks:
            try:
                hook(self, **kwargs)
            except Exception as e:
                logger.error(f"Hook execution failed: {e}")

    def _log_result_summary(self):
        """Log ETL result summary"""
        logger.info(f"ETL Result Summary for {self.job_name}:")
        logger.info(f"  Status: {self.result.status.value}")
        logger.info(f"  Records extracted: {self.result.records_extracted}")
        logger.info(f"  Records transformed: {self.result.records_transformed}")
        logger.info(f"  Records loaded: {self.result.records_loaded}")
        if self.result.errors:
            logger.info(f"  Errors: {len(self.result.errors)}")
            for error in self.result.errors[:5]:  # Show first 5 errors
                logger.info(f"    - {error}")

    # Hook registration methods
    def add_pre_extract_hook(self, hook: Callable):
        """Add pre-extract monitoring hook"""
        self._pre_extract_hooks.append(hook)

    def add_post_extract_hook(self, hook: Callable):
        """Add post-extract monitoring hook"""
        self._post_extract_hooks.append(hook)

    def add_pre_transform_hook(self, hook: Callable):
        """Add pre-transform monitoring hook"""
        self._pre_transform_hooks.append(hook)

    def add_post_transform_hook(self, hook: Callable):
        """Add post-transform monitoring hook"""
        self._post_transform_hooks.append(hook)

    def add_pre_load_hook(self, hook: Callable):
        """Add pre-load monitoring hook"""
        self._pre_load_hooks.append(hook)

    def add_post_load_hook(self, hook: Callable):
        """Add post-load monitoring hook"""
        self._post_load_hooks.append(hook)
