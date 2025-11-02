import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_GRU(nn.Module):
    """
    Convolutional Recurrent Neural Network for multi-label stuttering detection.
    Input: (batch, 1, 80, 298)  Log-Mel spectrogram
    Output: (batch, 6)  Multi-label logits (before sigmoid)
    """

    def __init__(self, num_classes=6, gru_hidden=128, num_gru_layers=2, dropout=0.3):
        super(CRNN_GRU, self).__init__()

        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # --- GRU ---
        # After 3 pools of (2,2), Mel dimension (80) → 10, Time dimension (298) → 37
        cnn_output_dim = 128 * 10  # channels × mel_height
        self.gru = nn.GRU(
            input_size=cnn_output_dim,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # --- Fully connected classifier ---
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, 80, 298)
        x = self.cnn(x)
        # reshape for GRU: (batch, time, features)
        b, c, h, t = x.size()  # (batch, 128, 10, 37)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time=37, channels=128, height=10)
        x = x.view(b, t, c * h)  # (batch, 37, 1280)
        # GRU
        gru_out, _ = self.gru(x)  # (batch, time, 2*hidden)
        # mean pooling over time
        x = torch.mean(gru_out, dim=1)
        # fully connected
        out = self.fc(x)  # (batch, num_classes)
        return out

if __name__ == "__main__":
    # quick sanity check
    model = CRNN_GRU(num_classes=6)
    x = torch.randn(4, 1, 80, 298)
    y = model(x)
    print("Input:", x.shape)
    print("Output:", y.shape)