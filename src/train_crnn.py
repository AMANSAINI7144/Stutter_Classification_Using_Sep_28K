#!/usr/bin/env python3
# src/train_crnn_hybrid.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from models.crnn_model import CRNN_GRU
import torch.nn.functional as F

# ======================
# CONFIG
# ======================
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.getcwd()
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
LOG_PATH = os.path.join(BASE_DIR, "results", "training_log_hybrid.txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ======================
# DATASET
# ======================
class FeatureDataset(Dataset):
    def __init__(self, split):
        self.csv_path = os.path.join(FEATURES_DIR, split, f"mapping_{split}.csv")
        self.data = pd.read_csv(self.csv_path)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        clipid = str(row.ClipId)
        feature_path = os.path.join(FEATURES_DIR, self.split, f"{clipid}.npy")
        feature = np.load(feature_path)

        # Normalize feature (zero mean, unit variance)
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)
        feature = feature.unsqueeze(0)  # shape: (1, 80, 298)

        # Convert label severities to binary presence (0 -> 0, >0 -> 1)
        labels_raw = [
            row.label_block,
            row.label_prolong,
            row.label_soundrep,
            row.label_wordrep,
            row.label_interjection,
            row.label_no_stutter
        ]
        labels = torch.tensor([1.0 if (v and float(v) > 0.0) else 0.0 for v in labels_raw], dtype=torch.float32)

        return feature, labels


# ======================
# HYBRID LOSS: BCE + FOCAL
# ======================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class HybridLoss(nn.Module):
    def __init__(self, lambda_focal=0.5, pos_weight=None, device='cpu'):
        super(HybridLoss, self).__init__()
        # pos_weight: list or tensor of length num_classes (or None)
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float, device=device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=2.0, alpha=0.75)
        self.lambda_focal = lambda_focal

    def forward(self, inputs, targets):
        bce_part = self.bce(inputs, targets)
        focal_part = self.focal(inputs, targets)
        loss = bce_part + self.lambda_focal * focal_part
        return loss



# ======================
# UTILITIES
# ======================
def save_log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
    print(msg)

def save_checkpoint(model, optimizer, epoch, val_loss, best=False):
    filename = "best_model_hybrid.pth" if best else "last_epoch_hybrid.pth"
    path = os.path.join(MODEL_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss
    }, path)
    save_log(f"[CHECKPOINT] Saved -> {path}")

# ======================
# TRAINING UTILS
# ======================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    f1 = f1_score(all_targets > 0, all_preds > 0.5, average='macro')
    return total_loss / len(loader.dataset), f1


# ======================
# EARLY STOPPING
# ======================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.verbose:
                save_log(f"[EarlyStopping] No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False


# ======================
# MAIN TRAINING LOOP
# ======================
def main():
    train_ds = FeatureDataset("train")
    val_ds = FeatureDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CRNN_GRU(num_classes=6).to(DEVICE)
    criterion = HybridLoss(lambda_focal=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    save_log(f"[INIT] Starting training on {DEVICE}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        elapsed = time.time() - start
        log = (f"Epoch {epoch:03d}/{EPOCHS} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
               f"Time: {elapsed/60:.2f} min")
        save_log(log)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best=True)
            save_log(f"[INFO] ✅ New best model saved (Val Loss={val_loss:.4f})")
        else:
            save_checkpoint(model, optimizer, epoch, val_loss, best=False)

        if early_stopping.step(val_loss):
            save_log("[STOP] Early stopping triggered.")
            break

    save_log("[DONE] Training complete.")


if __name__ == "__main__":
    main()
