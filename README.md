# E-Ticaret Satın Alma Niyeti Tahminleme (Purchase Intention)

**Proje Amacı:** Müşterilerin geçmiş işlemsel verilerini (Online Retail Dataset) analiz ederek, önümüzdeki çeyrekte (3 aylık periyot) tekrar satın alma gerçekleştirme olasılıklarını tahmin makine öğrenmesi mimarisidir. İşletmelerin proaktif müşteri segmentasyonu yapmasını ve pazarlama bütçelerini verimli kullanmasını amaçlar.

## Kurulum & Yapılandırma

```bash
# Depoyu klonlayın ve proje dizinine geçin
git clone <repo-url>
cd online_retail

# Sanal ortam oluşturun ve aktifleştirin
python -m venv .venv
.venv\Scripts\activate  # Windows için
# source .venv/bin/activate  # Linux/MacOS için

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

## Veri ve Çalıştırma

1. [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) dosyasını indirip `data/` klasörüne `Online_Retail.xlsx` olarak kaydedin.
2. Optimize edilmiş pipeline'ı çalıştırmak için:

```bash
python experiment_2.py
```

## Proje Yapısı

```
online_retail/
├── src/
│   ├── preprocessing_2.py     # Veri temizleme & yapılandırma
│   ├── features_2.py          # Dinamik (Rolling) özellik mühendisliği (En etkili 22 metrik)
│   ├── modeling_2.py          # Walk-forward doğrulama ve model eğitimi
│   └── reporting_2.py         # Sonuçların raporlarının üretilmesi
├── experiment_2.py            # Tüm mimariyi sırasıyla çalıştırır
├── results/                   # Analiz, segment metrikleri ve model çıktıları
├── requirements.txt
└── README.md
```


