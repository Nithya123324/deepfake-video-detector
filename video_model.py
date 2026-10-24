import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViT_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=2, num_classes=2):
        super(ViT_BiLSTM, self).__init__()
        
        # Vision Transformer backbone
        from torchvision.models import ViT_B_16_Weights
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()  # remove default classifier
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=768,   # ViT embedding size
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten sequence for ViT
        x = x.view(batch_size * seq_len, C, H, W)
        vit_features = self.vit(x)  # (batch*seq_len, 768)
        
        # Reshape back to sequence
        vit_features = vit_features.view(batch_size, seq_len, -1)
        
        # BiLSTM
        lstm_out, _ = self.lstm(vit_features)  # (batch, seq_len, hidden*2)
        
        # Use last timestep
        last_out = lstm_out[:, -1, :]
        
        # Classification
        out = self.fc(last_out)
        return out
