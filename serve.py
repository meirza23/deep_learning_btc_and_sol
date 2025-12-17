import gradio as gr
import torch
import numpy as np
import joblib
import yfinance as yf
from model import GRUModel
import datetime
import matplotlib.pyplot as plt
import os
import requests
import pandas as pd

# --- İkonları İndir ---
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
                with open(filename, 'wb') as f: f.write(response.content)
            except: pass
download_icons()

# --- Analiz Fonksiyonu ---
def analyze_crypto(symbol):
    # Grafik stilini daha estetik yap
    plt.style.use('seaborn-v0_8-darkgrid') 
    fig = plt.figure(figsize=(10, 5))
    
    try:
        model_path = f"model_{symbol}.pth"
        input_scaler_path = f"scaler_input_{symbol}.pkl"
        target_scaler_path = f"scaler_target_{symbol}.pkl"
        
        if not os.path.exists(model_path): return fig, "⚠️ Önce train.py çalıştırın."

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model yapısı
        model = GRUModel(input_size=1, hidden_size=256, num_layers=2)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except: return fig, "⚠️ Model uyumsuz. train.py çalıştırın."

        scaler_input = joblib.load(input_scaler_path)
        scaler_target = joblib.load(target_scaler_path)
        model.eval()

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=150) 
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            # MultiIndex sorununu çöz
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs('Close', level=0, axis=1)
                except: df = df.xs('Adj Close', level=0, axis=1)
            
            # Sadece Close sütununu al
            if len(df.columns) > 1:
                # Eğer 'Close' string olarak varsa al, yoksa ilk sütunu al
                if 'Close' in df.columns: df = df[['Close']]
                else: df = df.iloc[:, 0:1]
            
            df.columns = ['Close']
            
            if len(df) < 30: return fig, "Yetersiz veri."
            
            values = df.values.astype(float).flatten()
            last_seq = values[-30:].reshape(-1, 1) # Son 30 gün
            current_price = values[-1]

        except Exception as e: return fig, f"Veri hatası: {e}"

        # Tahmin İşlemi
        input_scaled = scaler_input.transform(last_seq)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            prediction_diff_scaled = model(input_tensor)
            
        pred_diff = scaler_target.inverse_transform(prediction_diff_scaled.numpy())
        predicted_change = float(pred_diff.item())
        predicted_price = round(current_price + predicted_change, 2)

        # Grafik Çizimi
        plot_data = values[-90:] 
        x_range = range(len(plot_data))
        
        plt.plot(x_range, plot_data, label='Son 90 Gün', color='#2980b9', linewidth=2.5)
        plt.scatter(len(plot_data), predicted_price, color='#e74c3c', s=120, label='AI Tahmini', zorder=5, edgecolor='white')
        
        # Son noktadan tahmine kesikli çizgi
        plt.plot([len(plot_data)-1, len(plot_data)], [plot_data[-1], predicted_price], 
                 color='#e74c3c', linestyle='--', alpha=0.8, linewidth=1.5)
        
        plt.title(f"{symbol} AI Fiyat Tahmini", fontsize=14, pad=15, fontweight='bold')
        plt.legend(frameon=True, fancybox=True, framealpha=1)
        plt.tight_layout()
        
        # Rapor Oluşturma
        yuzde = (predicted_change / current_price) * 100
        icon = "🚀 YÜKSELİŞ BEKLENTİSİ" if predicted_change > 0 else "🔻 DÜŞÜŞ BEKLENTİSİ"
        color_code = "green" if predicted_change > 0 else "red"
        fark_str = f"+${predicted_change:.2f}" if predicted_change > 0 else f"-${abs(predicted_change):.2f}"
        
        report = (
            f"ANALİZ RAPORU: {symbol}\n"
            f"──────────────────────────\n"
            f"💵 Mevcut Fiyat:   ${current_price:.2f}\n"
            f"🎯 AI Hedef Fiyat: ${predicted_price}\n"
            f"📊 Beklenen Fark:  {fark_str} (%{yuzde:.2f})\n"
            f"──────────────────────────\n"
            f"\n{icon}"
        )
        return fig, report

    except Exception as e: return fig, f"Sistem Hatası: {str(e)}"

def click_btc(): return analyze_crypto("BTC-USD")
def click_sol(): return analyze_crypto("SOL-USD")

# --- Arayüz Tasarımı (CSS ve Layout) ---

# CSS ile sayfa ortalama ve estetik dokunuşlar
custom_css = """
#main-container {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}
h1 {
    text-align: center; 
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: #2c3e50;
    margin-bottom: 5px;
}
.gradio-container {
    background-color: #f9fafb;
}
footer {visibility: hidden !important;}
"""

# Tema seçimi
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

# Düzeltme 1: css ve theme buradan kaldırıldı, launch'a taşındı.
with gr.Blocks(title="Crypto AI") as demo:
    
    with gr.Column(elem_id="main-container"):
        
        gr.Markdown("# 🤖 Kripto Para Yapay Zeka Tahmin Modeli")
        
        with gr.Row():
            btn_btc = gr.Button("Bitcoin Analiz Et", icon="btc_logo.png", variant="primary")
            btn_sol = gr.Button("Solana Analiz Et", icon="sol_logo.png", variant="primary")
            
        gr.Markdown("---") 
        
        with gr.Row():
            with gr.Column(scale=2): 
                plot_output = gr.Plot(label="Fiyat Grafiği")
            with gr.Column(scale=1): 
                # Düzeltme 2: show_copy_button=True kaldırıldı
                text_output = gr.Textbox(label="AI Raporu", lines=8)

    # Buton Aksiyonları
    btn_btc.click(click_btc, None, [plot_output, text_output])
    btn_sol.click(click_sol, None, [plot_output, text_output])

if __name__ == "__main__":
    # Düzeltme 1 (Devamı): css ve theme buraya eklendi
    demo.launch(share=True, css=custom_css, theme=theme)