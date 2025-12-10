import gradio as gr
import torch
import numpy as np
import joblib
import yfinance as yf
from model import LSTMModel
import datetime
import matplotlib.pyplot as plt
import os
import requests
import pandas as pd

def download_icons():
    icons = {
        "btc_logo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Bitcoin.svg/64px-Bitcoin.svg.png",
        "sol_logo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Solana_cryptocurrency_two.jpg/64px-Solana_cryptocurrency_two.jpg"
    }
    for filename, url in icons.items():
        if not os.path.exists(filename):
            try:
                headers = {'User-Agent': 'Mozilla/5.0'} 
                response = requests.get(url, headers=headers)
                with open(filename, 'wb') as f:
                    f.write(response.content)
            except: pass

download_icons()

def analyze_crypto(symbol):
    fig = plt.figure(figsize=(10, 5))
    
    try:
        model_path = f"model_{symbol}.pth"
        scaler_path = f"scaler_{symbol}.pkl"
        
        if not os.path.exists(model_path):
             plt.text(0.5, 0.5, f"HATA: {symbol} Eğitilmemiş!", ha='center', color='red')
             return fig, "⚠️ Önce train.py çalıştırın."

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size=1, hidden_size=128, num_layers=2) # 128 Uyumlu
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except RuntimeError:
            return fig, "⚠️ Model uyumsuzluğu. train.py tekrar çalıştırın."

        scaler = joblib.load(scaler_path)
        model.eval()

        # SEQ_LENGTH 60 olduğu için en az 60 gün veri çekiyoruz
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=120) 
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs('Close', level=0, axis=1)
            elif 'Close' in df.columns:
                 df = df['Close']
            
            values = df.values.astype(float).flatten()

        except Exception as e:
            plt.text(0.5, 0.5, "Veri Hatası", ha='center', color='red')
            return fig, f"⚠️ Veri indirilemedi: {e}"
        
        if len(values) < 60:
            plt.text(0.5, 0.5, "Yetersiz Veri!", ha='center', color='red')
            return fig, "⚠️ Yetersiz veri."

        # Son 60 günü al
        last_seq = values[-60:].reshape(-1, 1) 
        current_price = float(last_seq[-1][0])

        input_scaled = scaler.transform(last_seq)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            
        pred_actual = scaler.inverse_transform(prediction.numpy())
        predicted_price = round(float(pred_actual.item()), 2)

        # Grafik
        plot_data = values[-90:] 
        
        plt.plot(range(len(plot_data)), plot_data, label='Geçmiş Veri', color='#1f77b4', linewidth=2)
        plt.scatter(len(plot_data), predicted_price, color='red', s=100, label='AI Tahmini', zorder=5)
        last_real_val = float(plot_data[-1])
        plt.plot([len(plot_data)-1, len(plot_data)], [last_real_val, predicted_price], color='red', linestyle='--', alpha=0.7)
        
        plt.title(f"{symbol} Analizi", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        degisim = predicted_price - current_price
        yuzde = (degisim / current_price) * 100
        
        icon = "🚀 YÜKSELİŞ" if degisim > 0 else "🔻 DÜŞÜŞ"
        fark_str = f"+${degisim:.2f}" if degisim > 0 else f"-${abs(degisim):.2f}"
        
        report = (
            f"✅ ANALİZ TAMAMLANDI: {symbol}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Fiyat: ${current_price:.2f}\n"
            f"🔮 Tahmin: ${predicted_price}\n"
            f"📊 Fark: {fark_str} (%{yuzde:.2f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Yön: {icon}"
        )
        
        return fig, report

    except Exception as e:
        plt.clf()
        plt.text(0.5, 0.5, f"Hata: {str(e)}", ha='center', color='red', fontsize=8)
        return fig, f"Sistem Hatası: {str(e)}"

# ARAYÜZ
custom_css = "footer {visibility: hidden !important;} .gradio-container {min-height: 0px !important;}"

def click_btc(): return analyze_crypto("BTC-USD")
def click_sol(): return analyze_crypto("SOL-USD")

with gr.Blocks(title="Kripto AI Terminal") as demo:
    gr.Markdown("# 🧠 Kripto Para Yapay Zeka Terminali")
    gr.Markdown("Bitcoin ve Solana için Optimize Edilmiş LSTM Modeli")
    
    with gr.Row():
        btn_btc = gr.Button("Bitcoin (BTC)", icon="btc_logo.png")
        btn_sol = gr.Button("Solana (SOL)", icon="sol_logo.png")

    with gr.Row():
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Grafik")
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Rapor", lines=10)

    btn_btc.click(click_btc, None, [plot_output, text_output])
    btn_sol.click(click_sol, None, [plot_output, text_output])

if __name__ == "__main__":
    demo.launch(share=True, css=custom_css)