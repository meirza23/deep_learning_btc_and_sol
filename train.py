import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import GRUModel
import joblib
import yfinance as yf
import datetime
import os
import time

COINS = ['BTC-USD', 'SOL-USD']
START_DATE = '2020-01-01'
END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
SEQ_LENGTH = 30      
EPOCHS = 150          
LR = 0.0001           

def train_coin_model(symbol):
    print(f"\n==========================================")
    print(f"🚀 {symbol} için DELTA (FARK) EĞİTİMİ Başlıyor...")
    print(f"==========================================")

    df = pd.DataFrame()
    for i in range(5):
        try:
            temp_df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if not temp_df.empty:
                df = temp_df
                break
            time.sleep(2)
        except: pass
    
    if df.empty: return

    try:
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs('Close', level=0, axis=1)
            except: df = df.xs('Adj Close', level=0, axis=1)
        if len(df.columns) > 1 and 'Close' in df.columns: df = df[['Close']]
        df.columns = ['Close']
    except: return

    df['Diff'] = df['Close'].diff()
    df.dropna(inplace=True)
    
    print(f"✅ Veri Hazır. Boyut: {len(df)}")

    data_input = df[['Close']].values

    data_target = df[['Diff']].values

    scaler_input = MinMaxScaler(feature_range=(0, 1))
    data_scaled_input = scaler_input.fit_transform(data_input)
    
    scaler_target = MinMaxScaler(feature_range=(-1, 1)) 
    data_scaled_target = scaler_target.fit_transform(data_target)

    train_size = int(len(data_scaled_input) * 0.90)
    
    train_x = data_scaled_input[:train_size]
    train_y = data_scaled_target[:train_size]
    
    test_x = data_scaled_input[train_size - SEQ_LENGTH:]
    test_y = data_scaled_target[train_size - SEQ_LENGTH:]

    test_actual_prices = data_input[train_size - SEQ_LENGTH:]

    def create_sequences(data_x, data_y, seq_length):
        xs, ys = [], []
        for i in range(len(data_x) - seq_length):
            x = data_x[i:i+seq_length]
            y = data_y[i+seq_length] 
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(train_x, train_y, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_x, test_y, SEQ_LENGTH)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = GRUModel(input_size=1, hidden_size=256, num_layers=2)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []

    print(f"🧠 Model 'Farkı Bulmayı' Öğreniyor ({EPOCHS} Epoch)...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_preds_diff = model(X_test) 

    pred_diffs = scaler_target.inverse_transform(test_preds_diff.numpy())

    real_diffs = scaler_target.inverse_transform(y_test.numpy())

    real_dir = np.sign(real_diffs)
    pred_dir = np.sign(pred_diffs)
    
    correct = np.sum(real_dir == pred_dir)
    dir_acc = (correct / len(real_dir)) * 100

    base_prices = test_actual_prices[SEQ_LENGTH:-1] 

    min_len = min(len(base_prices), len(pred_diffs))
    
    reconstructed_prices = base_prices[:min_len] + pred_diffs[:min_len]
    actual_prices_target = test_actual_prices[SEQ_LENGTH+1:][:min_len]

    mape = np.mean(np.abs((actual_prices_target - reconstructed_prices) / actual_prices_target)) * 100

    print(f"\n✅ {symbol} SONUÇLAR:")
    print(f"📉 Fiyat Hatası (MAPE): %{mape:.2f}")
    print(f"🧭 YÖN BAŞARISI       : %{dir_acc:.2f} (Kritik Değer)")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Loss', color='orange')
    plt.savefig(f"grafik_loss_{symbol}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices_target, label='Gerçek Fiyat', color='blue')
    plt.plot(reconstructed_prices, label='Tahmin (Fiyat + Fark)', color='red', linestyle='--')
    plt.title(f'{symbol} Yön Başarısı: %{dir_acc:.2f}')
    plt.legend()
    plt.savefig(f"grafik_tahmin_{symbol}.png")
    plt.close()

    torch.save(model.state_dict(), f"model_{symbol}.pth")
    joblib.dump(scaler_input, f"scaler_input_{symbol}.pkl")
    joblib.dump(scaler_target, f"scaler_target_{symbol}.pkl")
    print(f"💾 Dosyalar kaydedildi.\n")

if __name__ == "__main__":
    for coin in COINS:
        train_coin_model(coin)