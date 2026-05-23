"""Feature construction for detecting business-like behaviour on consumer cards.

The transaction files are large (~13M rows), therefore Polars lazy scans are used
instead of loading everything into pandas at once.
"""
from __future__ import annotations

from pathlib import Path
import polars as pl

BUSINESS_MCCS = ["7311", "7372", "5968", "4816", "7399", "5045", "7392"]


def add_transaction_features(path: Path, label: int, merchants: pl.DataFrame) -> pl.LazyFrame:
    """Read one transactions file and add reusable per-transaction columns.

    Deliberately excludes card_tier from modelling because it reveals the known
    card product (Business vs consumer), rather than behavioural propensity.
    """
    return (
        pl.scan_parquet(path)
        .join(merchants.lazy(), on="merchant_id", how="left", suffix="_merchant")
        .with_columns(
            pl.lit(label).alias("label"),
            pl.col("transaction_timestamp").dt.hour().alias("hour"),
            pl.col("transaction_date").dt.weekday().alias("weekday"),
            pl.col("transaction_date").dt.strftime("%Y-%m").alias("month"),
            (pl.col("channel") == "online").cast(pl.Int8).alias("is_online"),
            (pl.col("channel") == "POS").cast(pl.Int8).alias("is_pos"),
            (pl.col("country") != "Kazakhstan").cast(pl.Int8).alias("is_foreign"),
            (pl.col("mcc").is_in(BUSINESS_MCCS)).cast(pl.Int8).alias("is_business_mcc"),
            (pl.col("mcc") == "7311").cast(pl.Int8).alias("is_advertising"),
            (pl.col("mcc") == "7372").cast(pl.Int8).alias("is_software"),
            (pl.col("mcc") == "5968").cast(pl.Int8).alias("is_subscription_tools"),
            (pl.col("mcc") == "4816").cast(pl.Int8).alias("is_cloud_digital"),
        )
        .with_columns(
            (pl.col("weekday") >= 6).cast(pl.Int8).alias("is_weekend"),
            ((pl.col("hour") >= 9) & (pl.col("hour") < 18)).cast(pl.Int8).alias("is_workhour"),
            ((pl.col("hour") >= 22) | (pl.col("hour") < 6)).cast(pl.Int8).alias("is_night"),
        )
    )


def aggregate_cards(tx: pl.LazyFrame) -> pl.DataFrame:
    """Aggregate raw transactions into one interpretable behavioural row per card."""
    base = tx.group_by(["card_number", "label"]).agg(
        pl.len().alias("tx_count"),
        pl.col("transaction_amount_kzt").sum().alias("total_amount"),
        pl.col("transaction_amount_kzt").mean().alias("mean_amount"),
        pl.col("transaction_amount_kzt").median().alias("median_amount"),
        pl.col("transaction_amount_kzt").std().fill_null(0).alias("std_amount"),
        pl.col("transaction_amount_kzt").quantile(0.90).alias("p90_amount"),
        pl.col("transaction_amount_kzt").max().alias("max_amount"),
        pl.col("transaction_date").n_unique().alias("active_days"),
        pl.col("merchant_id").n_unique().alias("unique_merchants"),
        pl.col("mcc").n_unique().alias("unique_mcc"),
        pl.col("country").n_unique().alias("unique_tx_countries"),
        pl.col("merchant_country").n_unique().alias("unique_merchant_countries"),
        pl.col("is_online").mean().alias("online_ratio"),
        pl.col("is_pos").mean().alias("pos_ratio"),
        pl.col("tokenized").mean().alias("tokenized_ratio"),
        pl.col("is_recurring").mean().alias("recurring_ratio"),
        pl.col("recurring_capable").mean().alias("recurring_capable_ratio"),
        pl.col("is_foreign").mean().alias("foreign_ratio"),
        pl.col("is_business_mcc").mean().alias("business_mcc_ratio"),
        pl.col("is_advertising").mean().alias("advertising_ratio"),
        pl.col("is_software").mean().alias("software_ratio"),
        pl.col("is_subscription_tools").mean().alias("subscription_tools_ratio"),
        pl.col("is_cloud_digital").mean().alias("cloud_digital_ratio"),
        pl.col("is_weekend").mean().alias("weekend_ratio"),
        pl.col("is_workhour").mean().alias("workhour_ratio"),
        pl.col("is_night").mean().alias("night_ratio"),
    ).with_columns(
        (pl.col("tx_count") / pl.col("active_days")).alias("tx_per_active_day"),
        (pl.col("std_amount") / (pl.col("mean_amount") + 1)).alias("amount_cv"),
    )

    monthly = (
        tx.group_by(["card_number", "label", "month"])
        .agg(
            pl.len().alias("monthly_tx_count"),
            pl.col("transaction_amount_kzt").sum().alias("monthly_total_amount"),
        )
        .group_by(["card_number", "label"])
        .agg(
            pl.col("month").n_unique().alias("active_months"),
            pl.col("monthly_total_amount").mean().alias("avg_monthly_amount"),
            pl.col("monthly_total_amount").std().fill_null(0).alias("std_monthly_amount"),
            pl.col("monthly_tx_count").std().fill_null(0).alias("std_monthly_tx"),
            pl.col("monthly_tx_count").mean().alias("avg_monthly_tx"),
        )
        .with_columns(
            (pl.col("std_monthly_amount") / (pl.col("avg_monthly_amount") + 1)).alias("monthly_amount_cv"),
            (pl.col("std_monthly_tx") / (pl.col("avg_monthly_tx") + 1)).alias("monthly_tx_cv"),
        )
    )
    return base.join(monthly, on=["card_number", "label"], how="left").collect(engine="streaming")


def build_feature_table(consumer_path: Path, business_path: Path, merchants_path: Path, output_path: Path) -> pl.DataFrame:
    merchants = pl.read_parquet(merchants_path).select(
        "merchant_id", "merchant_country", "recurring_capable"
    )
    consumer = aggregate_cards(add_transaction_features(consumer_path, 0, merchants))
    business = aggregate_cards(add_transaction_features(business_path, 1, merchants))
    features = pl.concat([consumer, business], how="vertical").sort(["label", "card_number"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(output_path)
    return features
