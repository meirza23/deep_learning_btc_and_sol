import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        
        # Dropout'u tekrar 0.2 yaptık (Eski, başarılı ayar)
        # Hidden size 128 olarak kaldı (Güçlü beyin)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out