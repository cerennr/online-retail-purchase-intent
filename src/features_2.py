import os
import sys

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
import config


FEATURES_2_PATH = os.path.join(config.DATA_DIR, "modeling_features_2.csv")
CLEANED_DATA_2_PATH = os.path.join(config.DATA_DIR, "cleaned_retail_data_2.parquet")


def _safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def _build_one_cutoff(df, cutoff, horizon_days=90, min_observation_days=120):
    cutoff = pd.to_datetime(cutoff)
    obs_start = cutoff - pd.Timedelta(days=min_observation_days)
    pred_end = cutoff + pd.Timedelta(days=horizon_days)

    df_obs = df[(df["InvoiceDate"] <= cutoff) & (df["InvoiceDate"] >= obs_start)].copy()
    df_pred = df[(df["InvoiceDate"] > cutoff) & (df["InvoiceDate"] <= pred_end)].copy()

    if df_obs.empty or df_pred.empty:
        return pd.DataFrame()

    if "Is_Return" not in df_obs.columns:
        df_obs["Is_Return"] = df_obs["InvoiceNo"].astype(str).str.startswith("C").astype(int)
    df_obs["Abs_Quantity"] = df_obs["Quantity"].abs()
    df_obs["GrossSpend"] = df_obs.get(
        "Gross_Spend",
        pd.Series(np.where(df_obs["Is_Return"] == 0, df_obs["TotalSpend"], 0.0), index=df_obs.index),
    )
    df_obs["ReturnSpend"] = df_obs.get(
        "Return_Amount",
        pd.Series(np.where(df_obs["Is_Return"] == 1, df_obs["TotalSpend"].abs(), 0.0), index=df_obs.index),
    )
    df_obs["GrossQuantity"] = df_obs.get(
        "Gross_Quantity",
        pd.Series(np.where(df_obs["Is_Return"] == 0, df_obs["Quantity"].clip(lower=0), 0), index=df_obs.index),
    )
    df_obs["ReturnQuantity"] = df_obs.get(
        "Return_Quantity",
        pd.Series(np.where(df_obs["Is_Return"] == 1, df_obs["Quantity"].abs(), 0), index=df_obs.index),
    )
    df_success = df_obs[(df_obs["Quantity"] > 0) & (df_obs["Is_Return"] == 0)].copy()
    df_success["Days_To_Cutoff"] = (cutoff - df_success["InvoiceDate"]).dt.days

    all_groups = df_obs.groupby("CustomerID")
    success_groups = df_success.groupby("CustomerID")

    features = all_groups.agg(
        Net_Monetary=("TotalSpend", "sum"),
        Gross_Monetary=("GrossSpend", "sum"),
        Gross_Quantity=("GrossQuantity", "sum"),
        Avg_Basket_Size=("InvoiceNo", lambda x: len(x) / x.nunique()),
        Mean_Quantity_Per_Invoice=("GrossQuantity", lambda x: x.sum() / max(df_obs.loc[x.index, "InvoiceNo"].nunique(), 1)),
        Mean_Spend_Per_Invoice=("GrossSpend", lambda x: x.sum() / max(df_obs.loc[x.index, "InvoiceNo"].nunique(), 1)),
    ).reset_index()

    success_features = success_groups.agg(
        Recency=("InvoiceDate", lambda x: (cutoff - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Unique_Products=("StockCode", "nunique"),
        First_Purchase_Date=("InvoiceDate", "min"),
        Last_Purchase_Date=("InvoiceDate", "max"),
    ).reset_index()

    features = features.merge(success_features, on="CustomerID", how="left")
    features["Frequency"] = features["Frequency"].fillna(0)
    features["Unique_Products"] = features["Unique_Products"].fillna(0)
    features["Recency"] = features["Recency"].fillna(min_observation_days + 30)
    features["First_Purchase_Date"] = features["First_Purchase_Date"].fillna(obs_start)
    features["Last_Purchase_Date"] = features["Last_Purchase_Date"].fillna(obs_start)

    lifespan = (cutoff - pd.to_datetime(features["First_Purchase_Date"])).dt.days
    features["Lifetime_Days"] = lifespan.clip(lower=0)
    features["Customer_Age_Ratio"] = features["Recency"] / (lifespan + 1)
    features["Diversity_Score"] = features["Unique_Products"] / (features["Gross_Quantity"] + 1e-5)

    invoice_dates = df_success.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
    invoice_dates = invoice_dates.sort_values(["CustomerID", "InvoiceDate"])
    invoice_dates["Days_Between"] = invoice_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    interval_stats = invoice_dates.groupby("CustomerID")["Days_Between"].agg(
        Purchase_Interval_Mean="mean",
        Purchase_Interval_Std="std",
    ).reset_index()
    features = features.merge(interval_stats, on="CustomerID", how="left")
    features["Purchase_Interval_Mean"] = features["Purchase_Interval_Mean"].fillna(min_observation_days)
    features["Purchase_Interval_Std"] = features["Purchase_Interval_Std"].fillna(0)
    expected_interval = features["Purchase_Interval_Mean"]
    features["Recency_vs_Cadence"] = features["Recency"] / (expected_interval + 1)

    for days in [60, 90]:
        win = df_success[df_success["Days_To_Cutoff"] <= days]
        freq = win.groupby("CustomerID")["InvoiceNo"].nunique().reset_index(name=f"Freq_Last_{days}D")
        spend = win.groupby("CustomerID")["TotalSpend"].sum().reset_index(name=f"Spend_Last_{days}D")
        features = features.merge(freq, on="CustomerID", how="left")
        features = features.merge(spend, on="CustomerID", how="left")
        features[f"Freq_Last_{days}D"] = features[f"Freq_Last_{days}D"].fillna(0)
        features[f"Spend_Last_{days}D"] = features[f"Spend_Last_{days}D"].fillna(0)

    df_pred_purchases = df_pred[
        (df_pred["Quantity"] > 0) & (~df_pred["InvoiceNo"].astype(str).str.contains("C", na=False))
    ]
    active_customers = set(df_pred_purchases["CustomerID"].unique())
    features["Target"] = features["CustomerID"].isin(active_customers).astype(int)
    features["Cutoff_Date"] = cutoff.date().isoformat()
    features["Cutoff_Month"] = cutoff.month

    final_cols = [
        "CustomerID",
        "Cutoff_Date",
        "Cutoff_Month",
        "Recency",
        "Frequency",
        "Net_Monetary",
        "Gross_Monetary",
        "Gross_Quantity",
        "Unique_Products",
        "Avg_Basket_Size",
        "Mean_Quantity_Per_Invoice",
        "Mean_Spend_Per_Invoice",
        "Diversity_Score",
        "Purchase_Interval_Mean",
        "Purchase_Interval_Std",
        "Recency_vs_Cadence",
        "Freq_Last_60D",
        "Freq_Last_90D",
        "Spend_Last_60D",
        "Spend_Last_90D",
        "Lifetime_Days",
        "Customer_Age_Ratio",
        "Target",
    ]
    return features[final_cols]


def build_rolling_features_2(parquet_path=None, output_path=None, horizon_days=90):
    parquet_path = parquet_path or CLEANED_DATA_2_PATH
    output_path = output_path or FEATURES_2_PATH

    df = pd.read_parquet(parquet_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()

    first_cutoff = (min_date + pd.Timedelta(days=120)).normalize()
    last_cutoff = (max_date - pd.Timedelta(days=horizon_days)).normalize()
    cutoffs = list(pd.date_range(first_cutoff, last_cutoff, freq="ME"))
    if last_cutoff not in cutoffs:
        cutoffs.append(last_cutoff)
    cutoffs = sorted(set(cutoffs))

    frames = []
    for cutoff in cutoffs:
        frame = _build_one_cutoff(df, cutoff, horizon_days=horizon_days)
        if not frame.empty:
            frames.append(frame)
            print(
                f"[features_2] {cutoff.date()} -> {frame.shape[0]} musteri, "
                f"target oran={frame['Target'].mean():.3f}"
            )

    if not frames:
        raise ValueError("Rolling feature uretimi icin uygun cutoff bulunamadi.")

    features = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"[features_2] Kaydedildi -> {output_path} | shape={features.shape}")
    return features


if __name__ == "__main__":
    # preprocessing_2'nin ürettiği özel parquet dosyasının yolunu veriyoruz
    input_parquet = os.path.join(config.DATA_DIR, "cleaned_retail_data_2.parquet")
    output_csv = os.path.join(config.DATA_DIR, "modeling_features_2.csv")
    
    build_rolling_features_2(parquet_path=input_parquet, output_path=output_csv)
