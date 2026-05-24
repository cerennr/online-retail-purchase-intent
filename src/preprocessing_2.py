import os
import sys

import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))
import config


def clean_and_save_data_2(input_path=None, output_path=None, force=False):
    input_path = input_path or config.RAW_DATA_PATH
    output_path = output_path or os.path.join(config.DATA_DIR, "cleaned_retail_data_2.parquet")

    if os.path.exists(output_path) and not force:
        print(f"[preprocessing_2] Temiz veri mevcut, kullaniliyor -> {output_path}")
        return output_path

    print("[preprocessing_2] Excel okunuyor ve gelismis temizlik yapiliyor...")
    cols = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ]

    df = pd.read_excel(input_path, usecols=cols)
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str)
    df["Country"] = df["Country"].astype(str)
    df["Is_Return"] = df["InvoiceNo"].str.startswith("C").astype(int)

    df = df[df["UnitPrice"] > 0]

    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]
    df["Line_Amount"] = df["Quantity"].abs() * df["UnitPrice"]
    df["Gross_Spend"] = np.where(df["Is_Return"] == 0, df["Line_Amount"], 0.0)
    df["Return_Amount"] = np.where(df["Is_Return"] == 1, df["Line_Amount"], 0.0)
    df["Gross_Quantity"] = np.where(df["Is_Return"] == 0, df["Quantity"].clip(lower=0), 0)
    df["Return_Quantity"] = np.where(df["Is_Return"] == 1, df["Quantity"].abs(), 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"[preprocessing_2] Gelismis veri kaydedildi -> {output_path}")
    return output_path


if __name__ == "__main__":
    clean_and_save_data_2(force=True)
