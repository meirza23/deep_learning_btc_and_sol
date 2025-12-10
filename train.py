import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
import joblib
import yfinance as yf
import datetime
import os
import time

# --- ESKİ "SNIPER" AYARLARI (BTC ve SOL için En İyisi) ---
COINS = ['BTC-USD', 'SOL-USD'] # ETH çıkarıldı
START_DATE = '2020-01-01'
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

SEQ_LENGTH = 60      # Tekrar 60 güne döndük (En iyi sonuç buradaydı)
EPOCHS = 300         # 300 Tura geri döndük (Hassas öğrenme)
LR = 0.0001          # Düşük hızda, sindire sindire eğitim

def train_coin_model(symbol):
    print(f"\n==========================================")
    print(f"🎯 {symbol} için SNIPER EĞİTİMİ Başlıyor...")
    print(f"==========================================")
    
    df = pd.DataFrame()
    for i in range(5):
        try:
            df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                break
            time.sleep(2)
        except Exception as e:
            print(f"Veri hatası: {e}")
    
    if df.empty or len(df) < 200:
        print(f"❌ HATA: {symbol} verisi yetersiz.")
        return

    print(f"✅ {len(df)} günlük veri indirildi. İşleniyor...")
    
    # Çoklu index temizliği (Eğer varsa)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs('Close', level=0, axis=1)
    elif 'Close' in df.columns:
        df = df['Close']
        
    data = df.values.astype(float).reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * 0.95)
    train_data = data_scaled[:train_size]

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
    
    if len(X_train) == 0: return

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    
    # Model Kurulumu (model.py ile uyumlu 128 hidden)
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"🧠 Model eğitiliyor ({EPOCHS} Epoch)... Kahveni al, bu biraz sürecek.")
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), f"model_{symbol}.pth")
    joblib.dump(scaler, f"scaler_{symbol}.pkl")
    print(f"💾 {symbol} modeli kusursuz şekilde kaydedildi.\n")

if __name__ == "__main__":
    for coin in COINS:
        train_coin_model(coin)
    print("\n🎉 EĞİTİM TAMAMLANDI! Artık serve.py çalıştırabilirsin.")