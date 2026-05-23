# Results from the supplied dataset

## Hold-out evaluation

The pipeline was run on the supplied parquet files using a stratified 80/20 card-level split. It excluded `card_tier`, card identifiers and raw merchant identifiers from modelling.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | False positives | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 1.0000 | 0.9996 | 0.9998 | 1.0000 | 1.0000 | 0 | 2 |
| Random Forest | 1.0000 | 0.9994 | 0.9997 | 1.0000 | 1.0000 | 0 | 3 |
| Logistic Regression | 0.9992 | 1.0000 | 0.9996 | 1.0000 | 1.0000 | 4 | 0 |

**Recommended final model:** CatBoost. It has the strongest F1 score among tied top PR-AUC models and is appropriate for nonlinear tabular behaviour patterns.

## Leading CatBoost features

| Feature | Importance |
|---|---:|
| `tokenized_ratio` | 30.75 |
| `business_mcc_ratio` | 15.22 |
| `weekend_ratio` | 11.97 |
| `recurring_ratio` | 8.44 |
| `pos_ratio` | 8.02 |
| `advertising_ratio` | 6.82 |
| `online_ratio` | 4.85 |

These drivers are interpretable: business-like cards exhibit distinct digital-payment, commercial-category, recurrence and weekly scheduling patterns.

## The important caution for the presentation

Performance is extremely high. Do **not** present this as proof of guaranteed performance on real customers. The files are synthetic and the known business versus consumer segments are strongly separated. In the final presentation, say:

> The model separates synthetic known business behaviour from synthetic consumer behaviour very accurately. For real deployment, the score should first be used as an outreach/ranking tool, and pilot campaign outcomes should be collected as true labels for model recalibration.

This is a stronger and more credible interpretation than claiming that the model has conclusively discovered hidden entrepreneurs.

## Consumer propensity distribution

After fitting CatBoost on all labelled proxy observations, the model ranked the consumer population. The consumer propensity distribution was highly concentrated near zero:

| Percentile | Score |
|---|---:|
| Median | 0.000019 |
| 95th | 0.000900 |
| 99th | 0.006964 |
| 99.5th | 0.016348 |
| 99.9th | 0.094840 |
| Maximum | 0.941106 |

The output `top_1000_consumer_candidates.csv` contains the highest-ranked candidates for follow-up analysis or a proposed business-product campaign.
