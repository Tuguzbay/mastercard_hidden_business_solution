"""Train, evaluate and explain behavioural classifiers."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, classification_report
)
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

EXCLUDED_COLUMNS = ["card_number", "label"]


def metrics_for(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(x_test).astype(int)
    proba = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def train_and_evaluate(feature_path: Path, output_dir: Path, random_state: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    data = pd.read_parquet(feature_path)
    x = data.drop(columns=EXCLUDED_COLUMNS)
    y = data["label"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=random_state
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=100, max_depth=16, min_samples_leaf=3,
                class_weight="balanced", n_jobs=2, random_state=random_state
            )),
        ]),
        "CatBoost": CatBoostClassifier(
            iterations=250, depth=7, learning_rate=0.07,
            loss_function="Logloss", eval_metric="AUC",
            auto_class_weights="Balanced", random_seed=random_state,
            verbose=False, allow_writing_files=False, thread_count=2
        ),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        if name == "CatBoost":
            model.fit(x_train, y_train, eval_set=(x_test, y_test), early_stopping_rounds=40, verbose=False)
        else:
            model.fit(x_train, y_train)
        fitted[name] = model
        results.append(metrics_for(name, model, x_test, y_test))

    metrics = pd.DataFrame(results).sort_values(["pr_auc", "f1"], ascending=False)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    best_name = metrics.iloc[0]["model"]
    best = fitted[best_name]
    pred = best.predict(x_test).astype(int)
    proba = best.predict_proba(x_test)[:, 1]

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model: {best_name}\n\n")
        f.write(classification_report(y_test, pred, target_names=["Consumer", "Business proxy"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Consumer", "Business proxy"], ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(plot_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title(f"ROC curve — {best_name}")
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
    ax.set_title(f"Precision–Recall curve — {best_name}")
    fig.tight_layout()
    fig.savefig(plot_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    catboost_model = fitted["CatBoost"]
    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": catboost_model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "catboost_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    top = importance.head(15).sort_values("importance")
    ax.barh(top["feature"], top["importance"])
    ax.set_title("Top 15 CatBoost behavioural features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(plot_dir / "feature_importance.png", dpi=180)
    plt.close(fig)

    # Business propensity scores: train on all labelled proxy observations and rank consumers.
    final_model = CatBoostClassifier(
        iterations=int(catboost_model.get_best_iteration() + 1) if catboost_model.get_best_iteration() > 0 else 300,
        depth=7, learning_rate=0.07, loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=random_state, verbose=False, allow_writing_files=False, thread_count=2
    )
    final_model.fit(x, y, verbose=False)
    consumers = data.loc[data["label"] == 0].copy()
    consumers["business_propensity_score"] = final_model.predict_proba(consumers[x.columns])[:, 1]
    consumers[["card_number", "business_propensity_score"]].sort_values(
        "business_propensity_score", ascending=False
    ).head(1000).to_csv(output_dir / "top_1000_consumer_candidates.csv", index=False)
    score_summary = consumers["business_propensity_score"].describe(percentiles=[.5, .9, .95, .99, .995, .999]).to_dict()
    with open(output_dir / "consumer_score_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in score_summary.items()}, f, indent=2)
