# Detecting Hidden Commercial Activity from Consumer Card Transactions

## Executive idea

This solution creates a **Business Propensity Score** for each consumer card. It does not attempt to predict from individual payments. Instead, it transforms six months of transactions into one behavioural profile per card, learns the patterns of known business cards, and ranks consumer cards whose behaviour is most business-like.

The framing matters: known `business_cards` are a **proxy label**, not direct evidence that a consumer is operating a hidden business. The final output should be used for targeted offers or review, not enforcement.

## Why this solution is competition-ready

- **Behavioural features rather than label leakage:** `card_tier` is excluded because it trivially reveals whether a supplied record came from a known Business-card product.
- **Scalable processing:** Polars lazy parquet aggregation is designed for the supplied ~12.8 million transactions.
- **Model comparison:** Logistic Regression baseline, Random Forest, and CatBoost.
- **Business-appropriate metrics:** confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC.
- **Explainability:** CatBoost feature importance plots and a documented feature dictionary.
- **Deliverable output:** ranked consumer candidates with `business_propensity_score`.

## Dataset snapshot from the provided files

| Segment | Cards | Transactions | Mean transaction (KZT) | Median transaction (KZT) | Online ratio | Recurring ratio | Tokenized ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Consumer | 80,000 | 9,832,487 | 54,045 | 11,892 | 46.51% | 2.72% | 38.63% |
| Known business | 25,000 | 2,997,593 | 156,535 | 77,224 | 84.67% | 13.34% | 60.00% |

These descriptive differences support the core hypothesis: known business-card activity is materially larger, more online-oriented, and more recurring than ordinary consumer activity.

## Project structure

```text
mastercard_hidden_business_solution/
├── run_pipeline.py
├── requirements.txt
├── README.md
├── src/
│   ├── build_features.py
│   └── train_models.py
├── docs/
│   ├── feature_dictionary.md
│   └── presentation_outline.md
├── data/
│   └── README.md
└── outputs/
    └── dataset_snapshot.csv
```

## Setup and execution

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the three parquet files in `data/`, using these names:

```text
data/consumer_cards_MDQ.parquet
data/business_cards_MDQ.parquet
data/merchants_reference.parquet
```

Run the full pipeline:

```bash
python run_pipeline.py \
  --consumer data/consumer_cards_MDQ.parquet \
  --business data/business_cards_MDQ.parquet \
  --merchants data/merchants_reference.parquet \
  --output-dir outputs
```

On PowerShell, write the same command on one line or use backticks instead of backslashes.

## Outputs produced by the pipeline

| File | Purpose |
|---|---|
| `card_features.parquet` | One behavioural feature vector per card. |
| `model_metrics.csv` | Comparison of baseline and advanced models. |
| `classification_report.txt` | Precision/recall/F1 summary of the best model. |
| `plots/confusion_matrix.png` | Required classifier evaluation visual. |
| `plots/roc_curve.png` | Discrimination across thresholds. |
| `plots/precision_recall_curve.png` | Relevant when selecting outreach candidates. |
| `catboost_feature_importance.csv` and plot | Model explanation. |
| `top_1000_consumer_candidates.csv` | Consumer cards ranked for follow-up. |
| `consumer_score_summary.json` | Score distribution for threshold planning. |

## Recommended presentation narrative

1. **Problem:** Some consumer cards show business-like operations, creating a missed cross-sell and service opportunity.
2. **Target definition:** Learn a proxy of business-like transaction behaviour from known business cards; rank consumer candidates.
3. **Leakage control:** Do not use `card_tier`, even though it would improve test metrics, because it defeats the purpose of detection.
4. **Feature story:** value, digital channel, recurrence, MCC intent, timing and monthly persistence.
5. **Model comparison:** establish a transparent baseline, then show CatBoost performance.
6. **Decision layer:** use propensity thresholds based on desired precision/recall and outreach capacity.
7. **Responsible use:** candidates should receive business-product offers or review, not automatic adverse treatment.

## What to improve before submission

- Run the full pipeline and paste the actual metrics and plots into the deck.
- Add SHAP values if time permits; feature importance is already included as a reproducible fallback.
- Evaluate candidate thresholds such as top 0.5%, 1% and 5% of consumers.
- Add a “three example candidate profiles” slide explaining why each high-score consumer was flagged.
- State explicitly that the synthetic dataset may contain unusually clean patterns and real deployment would require labelled follow-up outcomes.
