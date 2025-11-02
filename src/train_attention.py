#!/usr/bin/env python3
# src/train_attention.py
"""
Train script for CRNN_Attention with similar training recipe as Hybrid:
 - WeightedRandomSampler (softened)
 - Hybrid BCEWithLogits + Focal loss
 - pos_weight capped
 - logs/results saved to results_2/
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from models.crnn_attention import CRNN_Attention
import torch.nn.functional as F
import pandas as pd

# ======================
# CONFIG
# ======================
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
PATIENCE = 5
MAX_NORM = 5.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints", "attention_expt1")
RESULTS_DIR = os.path.join(BASE_DIR, "results_2")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log_attention.txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "experiments"), exist_ok=True)

POS_WEIGHT_CAP = 3.0
SOFTEN_SAMPLER = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# HYBRID LOSS using BCEWithLogits + focal
# ======================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class HybridLoss(nn.Module):
    def __init__(self, lambda_focal=0.5, pos_weight=None, device='cpu'):
        super().__init__()
        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float, device=device)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.lambda_focal = lambda_focal

    def forward(self, inputs, targets):
        return self.bce(inputs, targets) + self.lambda_focal * self.focal(inputs, targets)

# ======================
# DATASET
# ======================
class FeatureDataset(Dataset):
    def __init__(self, split):
        mapping_path = os.path.join(FEATURE_DIR, split, f"mapping_{split}.csv")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")
        df = pd.read_csv(mapping_path)
        # determine feature path column
        if "feature_path" in df.columns:
            self.paths = df["feature_path"].tolist()
        elif "rel_path" in df.columns:
            # rel_path is usually relative from features/<split>/<rel_path>
            self.paths = [os.path.join(FEATURE_DIR, split, p) for p in df["rel_path"].tolist()]
        elif "ClipId" in df.columns:
            self.paths = [os.path.join(FEATURE_DIR, split, f"{int(c)}.npy") for c in df["ClipId"].tolist()]
        else:
            raise KeyError("Mapping CSV must contain feature_path or rel_path or ClipId")
        self.label_cols = [
            "label_block",
            "label_prolong",
            "label_soundrep",
            "label_wordrep",
            "label_interjection",
            "label_no_stutter",
        ]
        labels = df[self.label_cols].fillna(0).values.astype(np.float32)
        # binarize: >0 -> 1
        self.labels = (labels > 0).astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.paths[idx])
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        x = torch.tensor(x, dtype=torch.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

# ======================
# UTILITIES
# ======================
def save_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def save_checkpoint(model, optimizer, epoch, val_loss, best=False):
    filename = "best_model_attention.pth" if best else f"epoch_{epoch:03d}_attention.pth"
    path = os.path.join(MODEL_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss
    }, path)
    save_log(f"[CHECKPOINT] Saved -> {path}")

# ======================
# TRAIN / VALIDATE
# ======================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(X)   # model should output logits (not sigmoid)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Val", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits, _ = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_true.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    f1 = f1_score(all_true > 0, all_preds > 0.5, average='macro', zero_division=0)
    return total_loss / len(loader.dataset), f1

# ======================
# MAIN LOOP
# ======================
def main():
    save_log(f"[INIT] Device={DEVICE} | epochs={EPOCHS} | batch={BATCH_SIZE}")

    train_ds = FeatureDataset("train")
    val_ds = FeatureDataset("val")

    # Build sampler (softened)
    try:
        label_cols = ['label_block', 'label_prolong', 'label_soundrep', 'label_wordrep', 'label_interjection', 'label_no_stutter']
        # try to find mapping CSV for train to compute sample weights
        mapping_path = os.path.join(FEATURE_DIR, "train", "mapping_train.csv")
        df_map = pd.read_csv(mapping_path)
        labels_arr = df_map[label_cols].fillna(0).astype(float).values
        labels_bin = (labels_arr > 0).astype(float)
        pos_counts = labels_bin.sum(axis=0) + 1e-9
        inv_label_freq = 1.0 / pos_counts
        sample_weights = (labels_bin * inv_label_freq[None, :]).sum(axis=1)
        sample_weights = np.where(sample_weights > 0, sample_weights, 0.1)
        if SOFTEN_SAMPLER:
            sample_weights = np.sqrt(sample_weights)
        sample_weights = sample_weights / np.mean(sample_weights)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
        save_log("[INFO] Using WeightedRandomSampler (softened)")
    except Exception as e:
        save_log(f"[WARN] Sampler failed ({e}), falling back to shuffle")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CRNN_Attention(num_classes=6).to(DEVICE)

    # load pos_weight and cap it
    pw_csv = os.path.join("results", "experiments", "label_diagnostics.csv")
    pos_weight_list = None
    if os.path.exists(pw_csv):
        try:
            df_pw = pd.read_csv(pw_csv, index_col=0)
            if "pos_weight" in df_pw.columns:
                pos_weight_list = df_pw["pos_weight"].tolist()
                save_log(f"[INFO] Loaded pos_weight from {pw_csv}: {pos_weight_list}")
        except Exception as e:
            save_log(f"[WARN] Could not read pos_weight CSV: {e}")

    if pos_weight_list is None:
        pos_weight_list = [1.357, 2.301, 4.026, 5.104, 1.921, 0.320]
        save_log(f"[INFO] Using fallback pos_weight: {pos_weight_list}")

    pos_weight_arr = np.minimum(np.array(pos_weight_list, dtype=float), POS_WEIGHT_CAP)
    pos_weight_list = pos_weight_arr.tolist()
    save_log(f"[INFO] Using capped pos_weight: {pos_weight_list}")

    criterion = HybridLoss(lambda_focal=0.5, pos_weight=pos_weight_list, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        elapsed = (time.time() - t0) / 60.0
        save_log(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {elapsed:.2f} min")

        # save last epoch checkpoint
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"last_epoch_{epoch:03d}_attention.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(MODEL_DIR, "best_model_attention.pth")
            torch.save(model.state_dict(), best_path)
            save_log(f"[CHECKPOINT] ✅ New best model saved -> {best_path}")
        else:
            patience_counter += 1
            save_log(f"[EarlyStopping] No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                save_log("[STOP] Early stopping triggered.")
                break

    save_log("[DONE] Training complete.")


if __name__ == "__main__":
    main()
