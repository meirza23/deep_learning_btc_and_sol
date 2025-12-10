# 🚀 Kripto Para Fiyat Tahmin Sistemi (LSTM)

Bu proje, Derin Öğrenme (Deep Learning) yöntemlerinden **LSTM (Long Short-Term Memory)** mimarisini kullanarak Bitcoin (BTC) ve Solana (SOL) kripto paralarının gelecek fiyat hareketlerini tahmin etmeyi amaçlar.

Proje, 2020'den günümüze kadar olan geçmiş fiyat verilerini **Yahoo Finance** üzerinden canlı çeker, eğitir ve bir web arayüzü üzerinden sunar.

## 📂 Proje Yapısı

* **`model.py`**: LSTM Yapay Sinir Ağı mimarisinin tanımlandığı dosya.
* **`train.py`**: Veri çekme, ön işleme, model eğitimi ve başarı grafiklerinin oluşturulduğu modül.
* **`serve.py`**: Gradio kütüphanesi ile oluşturulmuş, kullanıcı dostu web arayüzü.
* **`requirements.txt`**: Projenin çalışması için gerekli kütüphaneler.

## 🛠️ Kurulum

Projeyi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Modeli Eğitin:**
    ```bash
    python train.py
    ```
    *Bu işlem veri setini indirecek ve yapay zeka modellerini oluşturacaktır.*

3.  **Arayüzü Başlatın:**
    ```bash
    python serve.py
    ```
    *Terminalde verilen linke tıklayarak tarayıcınızda sistemi kullanabilirsiniz.*

## 📊 Model Performansı

Modelin başarısı **MAPE (Mean Absolute Percentage Error)** metriği ile ölçülmüştür.
* **Bitcoin (BTC):** ~%2.5 Hata Payı
* **Solana (SOL):** ~%3.2 Hata Payı

*(Detaylı grafikler proje klasöründe `grafik_tahmin_BTC-USD.png` dosyasında mevcuttur.)*

## 🧠 Kullanılan Teknolojiler

* **Dil:** Python 3.9+
* **Yapay Zeka:** PyTorch
* **Veri Analizi:** Pandas, NumPy, Scikit-learn
* **Görselleştirme:** Matplotlib
* **Arayüz:** Gradio
* **Veri Kaynağı:** Yahoo Finance API (yfinance)
