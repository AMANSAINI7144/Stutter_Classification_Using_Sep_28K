import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
#  CNN + Bi-GRU + Attention model for Stuttering Detection
# =====================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, pool=2, dropout=0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class AttentionLayer(nn.Module):
    """
    Simple attention pooling mechanism:
    computes weights over time frames and takes weighted sum.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, rnn_out):
        # rnn_out: [B, T, H]
        attn_weights = self.attention(rnn_out)   # [B, T, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # attention over time
        context = torch.sum(attn_weights * rnn_out, dim=1)  # weighted sum
        return context, attn_weights


class CRNN_Attention(nn.Module):
    def __init__(self,
                 n_mels=80,
                 num_classes=6,           # multi-label: 6 binary heads
                 cnn_dropout=0.25,
                 gru_hidden=256,
                 gru_layers=2,
                 bidirectional=True):
        super().__init__()

        # ---------------- CNN Backbone ---------------- #
        self.cnn = nn.Sequential(
            ConvBlock(1, 32, pool=2, dropout=cnn_dropout),     # -> [B,32,40,T/2]
            ConvBlock(32, 64, pool=2, dropout=cnn_dropout),    # -> [B,64,20,T/4]
            ConvBlock(64, 128, pool=2, dropout=cnn_dropout),   # -> [B,128,10,T/8]
            ConvBlock(128, 256, pool=2, dropout=cnn_dropout),  # -> [B,256,5,T/16]
        )

        # ---------------- GRU Encoder ---------------- #
        gru_input = 256 * (n_mels // 16)  # flatten freq dimension
        self.gru = nn.GRU(
            input_size=gru_input,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional,
        )

        self.attn = AttentionLayer(gru_hidden * (2 if bidirectional else 1))

        # ---------------- Classifier ---------------- #
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * (2 if bidirectional else 1), 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, time]
        x = self.cnn(x)  # [B, C, F, T]
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F]
        x = x.view(B, T, -1)  # flatten freq dim -> [B, T, C*F]
        rnn_out, _ = self.gru(x)  # [B, T, H*2]

        context, attn_weights = self.attn(rnn_out)  # [B, H*2]
        out = self.fc(context)  # [B, num_classes]
        out = torch.sigmoid(out)
        return out, attn_weights


# ---------------- Testing block ---------------- #
if __name__ == "__main__":
    model = CRNN_Attention()
    x = torch.randn(4, 1, 80, 298)  # batch=4
    y, attn = model(x)
    print("Output:", y.shape, "Attention:", attn.shape)
