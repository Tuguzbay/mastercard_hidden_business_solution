# PowerPoint outline: Hidden Commercial Activity Detection

## Slide 1 — Title
**From Consumer Transactions to Business Propensity**  
Detecting hidden commercial activity using explainable machine learning.

## Slide 2 — Business problem
- Some entrepreneurs operate through consumer cards.
- This can hide demand for business cards, acquiring, working-capital lending and merchant services.
- Objective: identify consumer cards whose behaviour resembles known business-card activity.

## Slide 3 — Crucial modelling choice
**We are predicting propensity, not declaring fraud or misconduct.**
- Training labels: known business vs consumer cardholders.
- Final target: business-like consumers for outreach/review.
- Limitation: proxy labels are not confirmed hidden-business labels.

## Slide 4 — Data overview
Show the dataset volumes and transaction period. Add a compact segment comparison:
- Consumer: 80,000 cards; 9.83M transactions.
- Business: 25,000 cards; 3.00M transactions.
- Merchant reference: 2,165 merchants.

## Slide 5 — Early evidence
Use a comparison chart:
- Mean transaction: 54,045 KZT vs 156,535 KZT.
- Online ratio: 46.5% vs 84.7%.
- Recurring ratio: 2.7% vs 13.3%.
- Tokenized ratio: 38.6% vs 60.0%.
Conclusion: there is a plausible behavioural signal to model.

## Slide 6 — Feature engineering
Show six feature families:
- volume/value
- recurrence
- channel behaviour
- merchant/MCC commercial intent
- schedule patterns
- monthly stability
Include: `card_tier` excluded to prevent product-label leakage.

## Slide 7 — Solution architecture
Transaction parquet files → merchant enrichment → card-level aggregation → train/test split → model comparison → CatBoost propensity score → top consumer candidates.

## Slide 8 — Models and evaluation
Compare:
- Logistic Regression: interpretable baseline.
- Random Forest: nonlinear reference.
- CatBoost: final tabular model.
Report confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC.

## Slide 9 — Results
Insert pipeline-generated:
- model metrics table
- confusion matrix
- PR curve
Explain trade-off: precision controls unnecessary outreach; recall controls missed business prospects.

## Slide 10 — Explainability
Insert feature importance plot or SHAP summary. Describe dominant signals in plain language.
Example wording: “The score increases when a consumer has high online operational spending and recurring business-tool behaviour across several months.”

## Slide 11 — Activation strategy
Rank consumers by Business Propensity Score:
- Top 0.5%: human/business banker review and tailored offers.
- Next 0.5–2%: digital business-product campaign.
- Monitor conversion and use outcomes to retrain a real hidden-entrepreneur model.

## Slide 12 — Limitations and next steps
- Synthetic data; performance may overestimate live accuracy.
- Known business-card labels are proxy targets.
- Avoid adverse actions from score alone.
- Next: pilot outreach, obtain response labels, retrain and calibrate thresholds.
