"""
ETL pipeline for calculating and loading daily market metrics
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from loguru import logger

from src.etl.base_etl import BaseETL
from src.db.snowflake_connector import SnowflakeConnector
from src.utils.config import Config


class MarketMetricsETL(BaseETL):
    """ETL pipeline for calculating daily market metrics (price + financial data)"""

    def __init__(self, config: Config):
        """
        Initialize Market Metrics ETL

        Args:
            config: Application configuration
        """
        # Create Snowflake connector
        snowflake_connector = SnowflakeConnector(config.snowflake)

        # Initialize base class
        super().__init__(
            job_name="market_metrics_etl",
            snowflake_connector=snowflake_connector,
            fmp_client=None,  # Not needed for metric calculations
            batch_size=config.app.batch_size,
            enable_monitoring=config.app.enable_monitoring,
        )

        # Store config for later use
        self.config = config

    def extract(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract daily price data joined with most recent financial data

        Args:
            symbols: Optional list of symbols to process. If None, process all.
            start_date: Optional start date for price data
            end_date: Optional end date for price data

        Returns:
            List of records combining daily prices with financial data
        """
        logger.info("Extracting data for market metrics calculation")

        # Simplified query using pre-calculated TTM values
        query = """
        WITH daily_prices AS (
            -- Get daily price data
            SELECT 
                dp.price_key,
                dp.company_key,
                dp.date_key,
                dd.date AS price_date,
                dc.symbol,
                dc.sector,
                dp.close_price,
                dp.volume
            FROM ANALYTICS.FACT_DAILY_PRICES dp
            JOIN ANALYTICS.DIM_DATE dd ON dp.date_key = dd.date_key
            JOIN ANALYTICS.DIM_COMPANY dc ON dp.company_key = dc.company_key
            WHERE 1=1
        ),
        latest_quarterly AS (
            -- Get most recent quarterly financial data as of each price date
            SELECT DISTINCT
                p.price_key,
                p.company_key,
                p.date_key,
                p.price_date,
                FIRST_VALUE(f.financial_key) OVER (
                    PARTITION BY p.price_key
                    ORDER BY f.accepted_date DESC
                ) as financial_key
            FROM daily_prices p
            JOIN ANALYTICS.FACT_FINANCIALS f
                ON p.company_key = f.company_key
                AND f.accepted_date <= p.price_date
                AND f.period_type IN ('Q1', 'Q2', 'Q3', 'Q4')
        ),
        latest_ttm AS (
            -- Get most recent TTM calculation as of each price date
            SELECT DISTINCT
                p.price_key,
                p.company_key,
                p.date_key,
                p.price_date,
                FIRST_VALUE(t.ttm_key) OVER (
                    PARTITION BY p.price_key
                    ORDER BY t.accepted_date DESC
                ) as ttm_key
            FROM daily_prices p
            JOIN ANALYTICS.FACT_FINANCIALS_TTM t
                ON p.company_key = t.company_key
                AND t.accepted_date <= p.price_date
        )
        SELECT DISTINCT
            p.price_key,
            p.company_key,
            p.date_key,
            p.price_date,
            p.symbol,
            p.sector,
            -- Price data
            p.close_price,
            p.volume,
            -- Quarterly financial data
            lq.financial_key,
            f.period_type,
            f.accepted_date as quarterly_accepted_date,
            f.eps_diluted as quarterly_eps_diluted,
            f.total_equity,
            f.total_assets,
            f.total_debt,
            f.cash_and_equivalents,
            f.net_debt,
            f.shares_outstanding as quarterly_shares_outstanding,
            f.operating_income as quarterly_operating_income,
            f.dividends_paid as quarterly_dividends_paid,
            -- Financial ratios
            fr.revenue_per_share as quarterly_revenue_per_share,
            fr.book_value_per_share,
            -- TTM financial data
            lt.ttm_key,
            t.accepted_date as ttm_accepted_date,
            t.ttm_revenue,
            t.ttm_net_income,
            t.ttm_eps_diluted,
            t.ttm_operating_income,
            t.ttm_operating_cash_flow,
            t.ttm_dividends_paid,
            t.latest_shares_outstanding as ttm_shares_outstanding,
            t.latest_total_equity as ttm_total_equity,
            t.latest_total_assets as ttm_total_assets,
            t.latest_total_debt as ttm_total_debt,
            t.latest_cash_and_equivalents as ttm_cash_and_equivalents,
            t.latest_net_debt as ttm_net_debt,
            t.quarters_included
        FROM daily_prices p
        LEFT JOIN latest_quarterly lq 
            ON p.price_key = lq.price_key
        LEFT JOIN ANALYTICS.FACT_FINANCIALS f 
            ON lq.financial_key = f.financial_key
        LEFT JOIN ANALYTICS.FACT_FINANCIAL_RATIOS fr
            ON lq.financial_key = fr.financial_key
        LEFT JOIN latest_ttm lt 
            ON p.price_key = lt.price_key
        LEFT JOIN ANALYTICS.FACT_FINANCIALS_TTM t 
            ON lt.ttm_key = t.ttm_key
        WHERE 1=1
        """

        params = []

        # Add symbol filter if provided
        if symbols:
            placeholders = ",".join(["%s" for _ in symbols])
            query += f" AND p.symbol IN ({placeholders})"
            params.extend(symbols)

        # Add date filters if provided
        if start_date:
            query += " AND p.price_date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND p.price_date <= %s"
            params.append(end_date)

        # Only process records that don't already have metrics calculated
        query += """
        AND NOT EXISTS (
            SELECT 1 
            FROM ANALYTICS.FACT_MARKET_METRICS mm
            WHERE mm.company_key = p.company_key
            AND mm.date_key = p.date_key
        )
        """

        # Only include records where we have financial data (either quarterly or TTM)
        query += " AND (lq.financial_key IS NOT NULL OR lt.ttm_key IS NOT NULL)"

        query += " ORDER BY p.symbol, p.price_date"

        try:
            records = self.snowflake.fetch_all(query, tuple(params) if params else None)
            logger.info(f"Extracted {len(records)} price records with financial data")

            # Log sample of data for debugging
            if records and len(records) > 0:
                sample = records[0]
                logger.debug(
                    f"Sample record - Symbol: {sample.get('SYMBOL')}, Date: {sample.get('PRICE_DATE')}, "
                    f"Quarterly data: {sample.get('FINANCIAL_KEY') is not None}, "
                    f"TTM data: {sample.get('TTM_KEY') is not None}, "
                    f"Quarters included: {sample.get('QUARTERS_INCLUDED', 0)}"
                )

            # Store metadata
            self.result.metadata["records_extracted"] = len(records)

            return records
        except Exception as e:
            logger.error(f"Failed to extract market data: {e}")
            self.result.errors.append(f"Extract failed: {str(e)}")
            raise

    def transform(
        self, raw_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate market metrics from price and financial data

        Args:
            raw_data: List of records combining price and financial data

        Returns:
            Dict with 'metrics' key containing calculated metric records
        """
        logger.info(f"Calculating market metrics for {len(raw_data)} records")

        metrics_data = []
        calculation_errors = 0

        for record in raw_data:
            try:
                # Base record
                metric_record = {
                    "company_key": record["COMPANY_KEY"],
                    "date_key": record["DATE_KEY"],
                    "financial_key": record.get(
                        "FINANCIAL_KEY"
                    ),  # May be None if only TTM data
                    "close_price": record["CLOSE_PRICE"],
                    "fiscal_period": (
                        record.get("PERIOD_TYPE", "TTM").upper()
                        if record.get("PERIOD_TYPE")
                        else "TTM"
                    ),
                }

                # Calculate market cap and enterprise value
                close_price = float(record.get("CLOSE_PRICE", 0) or 0)
                # Prefer TTM shares outstanding for consistency
                shares_outstanding = float(
                    record.get("TTM_SHARES_OUTSTANDING", 0)
                    or record.get("QUARTERLY_SHARES_OUTSTANDING", 0)
                    or 0
                )

                if close_price > 0 and shares_outstanding > 0:
                    market_cap = close_price * shares_outstanding
                    metric_record["market_cap"] = round(market_cap, 2)

                    # Enterprise Value = Market Cap + Total Debt - Cash
                    total_debt = float(
                        record.get("TTM_TOTAL_DEBT", 0)
                        or record.get("TOTAL_DEBT", 0)
                        or 0
                    )
                    cash = float(
                        record.get("TTM_CASH_AND_EQUIVALENTS", 0)
                        or record.get("CASH_AND_EQUIVALENTS", 0)
                        or 0
                    )
                    enterprise_value = market_cap + total_debt - cash
                    metric_record["enterprise_value"] = round(enterprise_value, 2)
                else:
                    metric_record["market_cap"] = None
                    metric_record["enterprise_value"] = None

                # P/E Ratio (Price to Earnings) - Use official EPS values
                quarterly_eps = float(record.get("QUARTERLY_EPS_DILUTED", 0) or 0)
                ttm_eps = float(record.get("TTM_EPS_DILUTED", 0) or 0)

                # Quarterly P/E
                if quarterly_eps > 0:
                    metric_record["pe_ratio"] = round(close_price / quarterly_eps, 2)
                else:
                    metric_record["pe_ratio"] = None

                # TTM P/E
                if ttm_eps > 0:
                    metric_record["pe_ratio_ttm"] = round(close_price / ttm_eps, 2)
                else:
                    metric_record["pe_ratio_ttm"] = None

                # P/B Ratio (Price to Book)
                book_value_per_share = float(record.get("BOOK_VALUE_PER_SHARE", 0) or 0)
                if book_value_per_share > 0:
                    metric_record["pb_ratio"] = round(
                        close_price / book_value_per_share, 2
                    )
                else:
                    metric_record["pb_ratio"] = None

                # P/S Ratio (Price to Sales) - Use pre-calculated revenue_per_share
                quarterly_revenue_per_share = float(
                    record.get("QUARTERLY_REVENUE_PER_SHARE", 0) or 0
                )

                # Quarterly P/S
                if quarterly_revenue_per_share > 0:
                    metric_record["ps_ratio"] = round(
                        close_price / quarterly_revenue_per_share, 2
                    )
                else:
                    metric_record["ps_ratio"] = None

                # TTM P/S
                ttm_revenue = float(record.get("TTM_REVENUE", 0) or 0)
                if ttm_revenue > 0 and shares_outstanding > 0:
                    ttm_revenue_per_share = ttm_revenue / shares_outstanding
                    metric_record["ps_ratio_ttm"] = round(
                        close_price / ttm_revenue_per_share, 2
                    )
                else:
                    metric_record["ps_ratio_ttm"] = None

                # EV/Revenue ratios
                if (
                    metric_record.get("enterprise_value")
                    and metric_record["enterprise_value"] > 0
                ):
                    # Quarterly EV/Revenue
                    if quarterly_revenue_per_share > 0 and shares_outstanding > 0:
                        quarterly_revenue = (
                            quarterly_revenue_per_share * shares_outstanding
                        )
                        metric_record["ev_to_revenue"] = round(
                            metric_record["enterprise_value"] / quarterly_revenue, 2
                        )
                    else:
                        metric_record["ev_to_revenue"] = None

                    # TTM EV/Revenue
                    if ttm_revenue > 0:
                        metric_record["ev_to_revenue_ttm"] = round(
                            metric_record["enterprise_value"] / ttm_revenue, 2
                        )
                    else:
                        metric_record["ev_to_revenue_ttm"] = None

                    # EV/EBITDA (using operating income as proxy)
                    ttm_operating_income = float(
                        record.get("TTM_OPERATING_INCOME", 0) or 0
                    )
                    if ttm_operating_income > 0:
                        metric_record["ev_to_ebitda"] = round(
                            metric_record["enterprise_value"] / ttm_operating_income, 2
                        )
                        metric_record["ev_to_ebit"] = metric_record[
                            "ev_to_ebitda"
                        ]  # Same since we don't have D&A
                    else:
                        metric_record["ev_to_ebitda"] = None
                        metric_record["ev_to_ebit"] = None
                else:
                    metric_record["ev_to_revenue"] = None
                    metric_record["ev_to_revenue_ttm"] = None
                    metric_record["ev_to_ebitda"] = None
                    metric_record["ev_to_ebit"] = None

                # Dividend metrics
                ttm_dividends = float(record.get("TTM_DIVIDENDS_PAID", 0) or 0)

                # Dividend Yield (using TTM dividends)
                if (
                    close_price > 0 and ttm_dividends < 0 and shares_outstanding > 0
                ):  # dividends are negative cash flows
                    dividends_per_share = abs(ttm_dividends) / shares_outstanding
                    metric_record["dividend_yield"] = round(
                        (dividends_per_share / close_price) * 100, 2
                    )
                else:
                    metric_record["dividend_yield"] = None

                # Payout Ratio
                ttm_net_income = float(record.get("TTM_NET_INCOME", 0) or 0)
                if (
                    ttm_net_income > 0 and ttm_dividends < 0
                ):  # dividends are negative cash flow
                    metric_record["payout_ratio"] = round(
                        (abs(ttm_dividends) / ttm_net_income) * 100, 2
                    )
                else:
                    metric_record["payout_ratio"] = None

                # PEG Ratio would require growth rates - skip for now
                metric_record["peg_ratio"] = None

                metrics_data.append(metric_record)

            except Exception as e:
                logger.warning(
                    f"Error calculating metrics for record {record.get('PRICE_KEY')}: {e}"
                )
                calculation_errors += 1

        # Store metadata
        self.result.metadata["records_transformed"] = len(metrics_data)
        self.result.metadata["calculation_errors"] = calculation_errors

        logger.info(f"Successfully calculated metrics for {len(metrics_data)} records")
        if calculation_errors > 0:
            logger.warning(
                f"Failed to calculate metrics for {calculation_errors} records"
            )

        return {"metrics": metrics_data}

    def load(self, transformed_data: Dict[str, List[Dict[str, Any]]]) -> int:
        """
        Load calculated metrics to FACT_MARKET_METRICS

        Args:
            transformed_data: Dict with 'metrics' key containing metric records

        Returns:
            Number of records loaded
        """
        metrics_data = transformed_data.get("metrics", [])

        if not metrics_data:
            logger.warning("No metric data to load")
            return 0

        logger.info(
            f"Loading {len(metrics_data)} metric records to FACT_MARKET_METRICS"
        )

        try:
            # Add timestamp
            current_timestamp = datetime.now(timezone.utc)
            for record in metrics_data:
                record["created_timestamp"] = current_timestamp

            # Insert into FACT_MARKET_METRICS
            self.snowflake.bulk_insert("ANALYTICS.FACT_MARKET_METRICS", metrics_data)

            logger.info(
                f"Successfully loaded {len(metrics_data)} records to FACT_MARKET_METRICS"
            )
            return len(metrics_data)

        except Exception as e:
            logger.error(f"Failed to load market metrics data: {e}")
            self.result.errors.append(f"Load failed: {str(e)}")
            raise

    def run(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete market metrics ETL pipeline

        Args:
            symbols: Optional list of symbols to process
            start_date: Optional start date for price data
            end_date: Optional end date for price data

        Returns:
            ETL result summary
        """
        logger.info(f"Starting {self.job_name} ETL pipeline")

        try:
            # Extract
            raw_data = self.extract(symbols, start_date, end_date)
            if not raw_data:
                logger.info("No new price data to process for market metrics")
                return {
                    "status": "success",
                    "records_processed": 0,
                    "message": "No new price data to process",
                }

            # Transform
            transformed_data = self.transform(raw_data)

            # Load
            records_loaded = self.load(transformed_data)

            return {
                "status": "success",
                "records_extracted": len(raw_data),
                "records_loaded": records_loaded,
                "calculation_errors": self.result.metadata.get("calculation_errors", 0),
            }

        except Exception as e:
            logger.error(f"Market metrics ETL pipeline failed: {e}")
            return {"status": "failed", "error": str(e), "records_processed": 0}
