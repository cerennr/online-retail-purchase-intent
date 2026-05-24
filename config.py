import os

# Projenin kök dizini (config.py'nin bulunduğu yer)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dizin Yapısı
DATA_DIR     = os.path.join(ROOT_DIR, "data")
RESULTS_DIR  = os.path.join(ROOT_DIR, "results")
SRC_DIR      = os.path.join(ROOT_DIR, "src")
MODELS_DIR   = os.path.join(ROOT_DIR, "models") # ── YENİ: Eğitilmiş ağırlıklar için klasör yolu ──

# Dosya Yolları
RAW_DATA_PATH          = os.path.join(DATA_DIR, "Online_Retail.xlsx")
CLEANED_DATA_PATH    = os.path.join(DATA_DIR, "cleaned_retail_data.parquet")
MODELING_FEATURES_PATH = os.path.join(DATA_DIR, "modeling_features.csv")

# Analiz Parametreleri
SPLIT_DATE = "2011-09-30"