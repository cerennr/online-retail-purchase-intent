import os
import pickle
import sys

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
import config
from features_2 import FEATURES_2_PATH
from modeling_2 import (
    FINAL_MODEL_2_PATH,
    WALKFORWARD_PREDICTIONS_2_PATH,
    WALKFORWARD_SUMMARY_2_PATH,
    binary_metrics_2,
    load_or_build_features_2,
    EXCLUDED_FEATURES,
)


BASELINE_RESULTS_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_baseline_comparison.csv")
BASELINE_SUMMARY_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_baseline_comparison_summary.csv")
FEATURE_IMPORTANCE_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_feature_importance.csv")
SEGMENT_METRICS_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_segment_metrics.csv")


def run_baseline_comparison_2(features_path=FEATURES_2_PATH):
    df = load_or_build_features_2(features_path, rebuild_features=False)
    rows = []

    for cutoff, fold in df.groupby("Cutoff_Date"):
        y_true = fold["Target"]
        rules = {
            "Active_Last_90D": (fold["Freq_Last_90D"] > 0).astype(int),
            "Active_Last_60D": (fold["Freq_Last_60D"] > 0).astype(int),
            "Recency_LTE_60": (fold["Recency"] <= 60).astype(int),
            "Frequency_GTE_2": (fold["Frequency"] >= 2).astype(int),
            "Top_35pct_By_Recency": (fold["Recency"].rank(method="first", ascending=True, pct=True) <= 0.35).astype(int),
        }

        for name, pred in rules.items():
            row = {
                "Baseline": name,
                "Cutoff_Date": cutoff.date().isoformat(),
                "Customers": len(fold),
                "Target_Rate": y_true.mean(),
            }
            row.update(binary_metrics_2(y_true, pred))
            rows.append(row)

    results = pd.DataFrame(rows)
    summary = _summarize_baselines(results)
    results.to_csv(BASELINE_RESULTS_2_PATH, index=False)
    summary.to_csv(BASELINE_SUMMARY_2_PATH, index=False)
    print(f"[reporting_2] Baseline summary -> {BASELINE_SUMMARY_2_PATH}")
    print(summary.to_string(index=False))
    return results, summary


def _summarize_baselines(results):
    return (
        results.groupby("Baseline", as_index=False)
        .agg(
            Folds=("Cutoff_Date", "nunique"),
            Mean_Macro_F1=("Macro_F1", "mean"),
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Recall_Class_0=("Recall_Class_0", "mean"),
            Mean_Recall_Class_1=("Recall_Class_1", "mean"),
            Mean_Precision_Class_1=("Precision_Class_1", "mean"),
        )
        .sort_values("Mean_Macro_F1", ascending=False)
    )


def export_feature_importance_2(features_path=FEATURES_2_PATH):
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in ["CustomerID", "Cutoff_Date", "Target"] + EXCLUDED_FEATURES]

    with open(FINAL_MODEL_2_PATH, "rb") as f:
        model = pickle.load(f)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.zeros(len(feature_cols))

    result = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    result.to_csv(FEATURE_IMPORTANCE_2_PATH, index=False)
    print(f"[reporting_2] Feature importance -> {FEATURE_IMPORTANCE_2_PATH}")
    return result


def export_segment_metrics_2(features_path=FEATURES_2_PATH):
    summary = pd.read_csv(WALKFORWARD_SUMMARY_2_PATH)
    best_model = summary.sort_values("Mean_Macro_F1", ascending=False).iloc[0]["Model"]

    features = pd.read_csv(features_path)
    features["Cutoff_Date"] = pd.to_datetime(features["Cutoff_Date"])
    preds = pd.read_csv(WALKFORWARD_PREDICTIONS_2_PATH)
    preds["Cutoff_Date"] = pd.to_datetime(preds["Cutoff_Date"])
    preds = preds[preds["Model"] == best_model].copy()

    merged = preds.merge(features, on=["CustomerID", "Cutoff_Date", "Target"], how="left")
    merged["Segment"] = merged.apply(_assign_segment, axis=1)

    rows = []
    for segment, part in merged.groupby("Segment"):
        rows.append(
            {
                "Model": best_model,
                "Segment": segment,
                "Customers": len(part),
                "Target_Rate": part["Target"].mean(),
                **binary_metrics_2(part["Target"], part["Prediction"]),
                "Avg_Purchase_Probability": part["Purchase_Probability"].mean(),
            }
        )

    result = pd.DataFrame(rows).sort_values("Customers", ascending=False)
    result.to_csv(SEGMENT_METRICS_2_PATH, index=False)
    print(f"[reporting_2] Segment metrics -> {SEGMENT_METRICS_2_PATH}")
    return result


def _assign_segment(row):
    if row["Frequency"] <= 1:
        return "New_or_One_Time"
    if row["Recency"] <= 30:
        return "Recently_Active"
    if row["Recency"] > 90:
        return "Dormant"
    return "Established"


def run_reporting_2(features_path=FEATURES_2_PATH):
    baselines = run_baseline_comparison_2(features_path)
    importance = export_feature_importance_2(features_path)
    segments = export_segment_metrics_2(features_path)
    print(importance.head(12).to_string(index=False))
    print(segments.to_string(index=False))
    return baselines, importance, segments


if __name__ == "__main__":
    run_reporting_2()
