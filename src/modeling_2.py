import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
import config
from features_2 import CLEANED_DATA_2_PATH, FEATURES_2_PATH, build_rolling_features_2
from preprocessing_2 import clean_and_save_data_2


WALKFORWARD_METRICS_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_walkforward_metrics.csv")
WALKFORWARD_SUMMARY_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_walkforward_summary.csv")
WALKFORWARD_PREDICTIONS_2_PATH = os.path.join(config.RESULTS_DIR, "experiment_2_walkforward_predictions.csv")
FINAL_MODEL_2_PATH = os.path.join(config.MODELS_DIR, "experiment_2_walkforward_best_model.pkl")
FINAL_SCALER_2_PATH = os.path.join(config.MODELS_DIR, "experiment_2_walkforward_scaler.pkl")


def load_or_build_features_2(features_path=FEATURES_2_PATH, rebuild_features=False):
    if not os.path.exists(CLEANED_DATA_2_PATH):
        clean_and_save_data_2()
    if rebuild_features or not os.path.exists(features_path):
        build_rolling_features_2(parquet_path=CLEANED_DATA_2_PATH, output_path=features_path)

    df = pd.read_csv(features_path)
    df["Cutoff_Date"] = pd.to_datetime(df["Cutoff_Date"])
    return df.sort_values(["Cutoff_Date", "CustomerID"]).reset_index(drop=True)


def binary_metrics_2(y_true, y_pred):
    return {
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Recall_Class_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall_Class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Precision_Class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def _model_specs():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            "scaled": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            "scaled": False,
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=1,
            ),
            "scaled": False,
        },
    }


def _best_threshold_by_macro_f1(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [(thr, f1_score(y_true, y_prob >= thr, average="macro", zero_division=0)) for thr in thresholds]
    return max(scores, key=lambda item: item[1])


def _precision_at_top_percent(y_true, y_prob, percent):
    n = max(1, int(np.ceil(len(y_true) * percent)))
    order = np.argsort(y_prob)[::-1][:n]
    return precision_score(np.asarray(y_true)[order], np.ones(n), zero_division=0)


def _lift_at_top_percent(y_true, y_prob, percent):
    baseline = np.mean(y_true)
    return 0 if baseline == 0 else _precision_at_top_percent(y_true, y_prob, percent) / baseline


def run_walkforward_validation_2(features_path=FEATURES_2_PATH, min_train_cutoffs=2, rebuild_features=False):
    df = load_or_build_features_2(features_path, rebuild_features=rebuild_features)
    cutoffs = sorted(df["Cutoff_Date"].unique())
    drop_cols = ["CustomerID", "Cutoff_Date", "Target"]
    rows, predictions = [], []

    for test_idx in range(min_train_cutoffs, len(cutoffs)):
        test_cutoff = cutoffs[test_idx]
        train_df = df[df["Cutoff_Date"].isin(cutoffs[:test_idx])].copy()
        test_df = df[df["Cutoff_Date"] == test_cutoff].copy()

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["Target"]
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df["Target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for name, spec in _model_specs().items():
            model = clone(spec["model"])
            xtr = X_train_scaled if spec["scaled"] else X_train
            xte = X_test_scaled if spec["scaled"] else X_test
            model.fit(xtr, y_train)

            train_prob = model.predict_proba(xtr)[:, 1]
            threshold, train_macro_f1 = _best_threshold_by_macro_f1(y_train, train_prob)
            test_prob = model.predict_proba(xte)[:, 1]
            test_pred = (test_prob >= threshold).astype(int)

            row = {
                "Model": name,
                "Test_Cutoff": pd.Timestamp(test_cutoff).date().isoformat(),
                "Train_Cutoffs": test_idx,
                "Train_Customers": len(train_df),
                "Test_Customers": len(test_df),
                "Target_Rate": y_test.mean(),
                "Train_Selected_Threshold": threshold,
                "Train_Macro_F1_At_Threshold": train_macro_f1,
                "AUC": roc_auc_score(y_test, test_prob),
                **binary_metrics_2(y_test, test_pred),
                "Precision_Top_10pct": _precision_at_top_percent(y_test, test_prob, 0.10),
                "Lift_Top_10pct": _lift_at_top_percent(y_test, test_prob, 0.10),
            }
            rows.append(row)

            fold_predictions = test_df[["CustomerID", "Cutoff_Date", "Target"]].copy()
            fold_predictions["Model"] = name
            fold_predictions["Purchase_Probability"] = test_prob
            fold_predictions["Selected_Threshold"] = threshold
            fold_predictions["Prediction"] = test_pred
            predictions.append(fold_predictions)

            print(
                f"[modeling_2] {name} | test={row['Test_Cutoff']} | "
                f"macro_f1={row['Macro_F1']:.3f} | bal_acc={row['Balanced_Accuracy']:.3f} | auc={row['AUC']:.3f}"
            )

    metrics = pd.DataFrame(rows)
    summary = _summarize_walkforward(metrics)
    metrics.to_csv(WALKFORWARD_METRICS_2_PATH, index=False)
    summary.to_csv(WALKFORWARD_SUMMARY_2_PATH, index=False)
    pd.concat(predictions, ignore_index=True).to_csv(WALKFORWARD_PREDICTIONS_2_PATH, index=False)
    _fit_and_save_final_model(df, summary.iloc[0]["Model"], drop_cols)

    print(f"[modeling_2] Walk-forward summary -> {WALKFORWARD_SUMMARY_2_PATH}")
    print(summary.to_string(index=False))
    return metrics, summary


def _summarize_walkforward(metrics):
    return (
        metrics.groupby("Model", as_index=False)
        .agg(
            Folds=("Test_Cutoff", "nunique"),
            Mean_AUC=("AUC", "mean"),
            Std_AUC=("AUC", "std"),
            Mean_Macro_F1=("Macro_F1", "mean"),
            Std_Macro_F1=("Macro_F1", "std"),
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Recall_Class_0=("Recall_Class_0", "mean"),
            Mean_Recall_Class_1=("Recall_Class_1", "mean"),
            Mean_Lift_Top_10pct=("Lift_Top_10pct", "mean"),
        )
        .sort_values("Mean_Macro_F1", ascending=False)
    )


def _fit_and_save_final_model(df, best_model_name, drop_cols):
    X_train = df.drop(columns=drop_cols)
    y_train = df["Target"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    spec = _model_specs()[best_model_name]
    model = clone(spec["model"])

    xtr = X_train_scaled if spec["scaled"] else X_train
    model.fit(xtr, y_train)

    with open(FINAL_MODEL_2_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(FINAL_SCALER_2_PATH, "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    run_walkforward_validation_2(rebuild_features=True)
