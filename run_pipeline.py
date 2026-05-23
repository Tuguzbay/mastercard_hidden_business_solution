"""Run the Mastercard hidden-commercial-activity solution end to end.

Example:
python run_pipeline.py --consumer data/consumer_cards_MDQ.parquet --business data/business_cards_MDQ.parquet --merchants data/merchants_reference.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
from src.build_features import build_feature_table
from src.train_models import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--business", type=Path, required=True)
    parser.add_argument("--merchants", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    feature_path = args.output_dir / "card_features.parquet"
    features = build_feature_table(args.consumer, args.business, args.merchants, feature_path)
    print(f"Built feature table: {features.height:,} cards x {features.width:,} columns")
    train_and_evaluate(feature_path, args.output_dir)
    print(f"Finished. Results saved to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
