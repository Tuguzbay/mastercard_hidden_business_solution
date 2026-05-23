# Feature dictionary

## Deliberately excluded features

| Column | Reason for exclusion |
|---|---|
| `card_tier` | Direct product-label leakage: the known business file contains `Business`, while the consumer file uses consumer tiers. It would make the model classify the existing product, not behaviour. |
| `card_number` | Identifier, not behaviour. Kept only for scoring output. |
| `bank_name` | Can create institution-specific artefacts rather than commercial-activity signals. |
| raw `merchant_id` | The model should generalise from behaviour and merchant categories, rather than memorising particular generated merchants. |

## Core behavioural features

| Group | Features | Business interpretation |
|---|---|---|
| Volume and value | `tx_count`, `total_amount`, `mean_amount`, `median_amount`, `std_amount`, `p90_amount`, `max_amount` | Higher turnover and larger purchase sizes may indicate operating spend. |
| Activity | `active_days`, `tx_per_active_day` | Frequency and operational regularity. |
| Diversity | `unique_merchants`, `unique_mcc`, `unique_tx_countries`, `unique_merchant_countries` | Breadth or concentration of business purchasing. |
| Channel | `online_ratio`, `pos_ratio`, `tokenized_ratio` | Digital-heavy operations and payment habits. |
| Recurrence | `recurring_ratio`, `recurring_capable_ratio` | SaaS/subscription-like operational spend. |
| Geography | `foreign_ratio` | Imported tools, advertising, cloud or cross-border services. |
| MCC-based proxies | `business_mcc_ratio`, `advertising_ratio`, `software_ratio`, `subscription_tools_ratio`, `cloud_digital_ratio` | Explainable commercial-intent indicators. |
| Timing | `weekend_ratio`, `workhour_ratio`, `night_ratio` | Operational schedule signals; interpret carefully because synthetic generation can affect timing. |
| Stability | `active_months`, `avg_monthly_amount`, `monthly_amount_cv`, `monthly_tx_cv` | Persistent activity is stronger evidence than one unusual month. |

## Important modelling caution

The supplied labels represent **known business cardholders versus ordinary consumer cardholders**. They are a proxy for the real target: hidden entrepreneurs among consumer cards. Accordingly, ranked consumer results should be positioned as **candidates for outreach or further review**, not confirmed undeclared commercial activity.
