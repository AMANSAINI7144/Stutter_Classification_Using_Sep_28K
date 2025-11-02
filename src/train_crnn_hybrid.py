#!/usr/bin/env python3
# src/train_crnn_hybrid.py
"""
Train script for CRNN_GRU (Hybrid) with:
 - WeightedRandomSampler (softened)
 - Hybrid BCEWithLogits + Focal loss
 - capped pos_weight to avoid extremes
 - results go to results_2/ to keep experiments separate
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from models.crnn_model import CRNN_GRU
import torch.nn.functional as F

# ======================
# CONFIG
# ======================
EPOCHS = 30               # change if you want longer/shorter runs
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 5              # early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.getcwd()
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR = os.path.join(BASE_DIR, "models", "checkpoints", "hybrid_expt1")
RESULTS_DIR = os.path.join(BASE_DIR, "results_2")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log_hybrid.txt")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "experiments"), exist_ok=True)

# pos_weight cap to prevent extremely large weights
POS_WEIGHT_CAP = 3.0
# sampler softening flag (square-root), set False to disable
SOFTEN_SAMPLER = True

# ======================
# DATASET
# ======================
class FeatureDataset(Dataset):
    def __init__(self, split):
        self.csv_path = os.path.join(FEATURES_DIR, split, f"mapping_{split}.csv")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Mapping CSV not found: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # ClipId or rel_path depending on mapping
        if "ClipId" in row.index and str(row.ClipId).strip() != "":
            clipid = str(row.ClipId)
            feature_path = os.path.join(FEATURES_DIR, self.split, f"{clipid}.npy")
        elif "rel_path" in row.index and str(row.rel_path).strip() != "":
            feature_path = os.path.join(FEATURES_DIR, self.split, row.rel_path)
        elif "feature_path" in row.index and str(row.feature_path).strip() != "":
            feature_path = row.feature_path
        else:
            raise KeyError("Could not determine feature path in mapping CSV. Expected ClipId or rel_path or feature_path.")

        feature = np.load(feature_path)
        # to tensor and normalize per-sample
        feature = torch.tensor(feature, dtype=torch.float32)
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)  # (1, n_mels, time)

        labels = torch.tensor([
            row.label_block,
            row.label_prolong,
            row.label_soundrep,
            row.label_wordrep,
            row.label_interjection,
            row.label_no_stutter
        ], dtype=torch.float32)

        # Convert annotation severities (0-3) into binary presence (0/1)
        # If values are 0..3, treat >0 as positive
        labels = (labels > 0).float()

        return feature, labels


# ======================
# HYBRID LOSS: BCEWithLogits + Focal
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
        return bce_part + self.lambda_focal * focal_part


# ======================
# UTILITIES
# ======================
def save_log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def save_checkpoint(model, optimizer, epoch, val_loss, best=False):
    filename = "best_model_hybrid.pth" if best else f"epoch_{epoch:03d}_hybrid.pth"
    path = os.path.join(MODEL_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": val_loss
    }, path)
    save_log(f"[CHECKPOINT] Saved -> {path}")


# ======================
# TRAIN / VALIDATION
# ======================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
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

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    f1 = f1_score(all_targets > 0, all_preds > 0.5, average='macro', zero_division=0)
    return total_loss / len(loader.dataset), f1


# ======================
# MAIN
# ======================
def main():
    save_log(f"[INIT] Device={DEVICE} | epochs={EPOCHS} | batch={BATCH_SIZE}")

    train_ds = FeatureDataset("train")
    val_ds = FeatureDataset("val")

    # Build soft WeightedRandomSampler
    try:
        label_cols = ['label_block', 'label_prolong', 'label_soundrep', 'label_wordrep', 'label_interjection', 'label_no_stutter']
        labels_arr = train_ds.data[label_cols].fillna(0).astype(float).values
        labels_bin = (labels_arr > 0).astype(float)
        pos_counts = labels_bin.sum(axis=0) + 1e-9
        inv_label_freq = 1.0 / pos_counts

        sample_weights = (labels_bin * inv_label_freq[None, :]).sum(axis=1)
        sample_weights = np.where(sample_weights > 0, sample_weights, 0.1)

        if SOFTEN_SAMPLER:
            sample_weights = np.sqrt(sample_weights)

        sample_weights = sample_weights / np.mean(sample_weights)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
        save_log("[INFO] Using WeightedRandomSampler (softened)")
    except Exception as e:
        save_log(f"[WARN] Sampler creation failed ({e}); falling back to shuffle.")
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CRNN_GRU(num_classes=6).to(DEVICE)

    # Load pos_weight and cap it
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

    # Cap pos_weight to avoid too-large weights
    pos_weight_arr = np.minimum(np.array(pos_weight_list, dtype=float), POS_WEIGHT_CAP)
    pos_weight_list = pos_weight_arr.tolist()
    save_log(f"[INFO] Using capped pos_weight: {pos_weight_list}")

    criterion = HybridLoss(lambda_focal=0.5, pos_weight=pos_weight_list, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_f1 = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        elapsed = (time.time() - t0) / 60.0
        log = f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {elapsed:.2f} min"
        save_log(log)

        # checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, best=True)
            patience = 0
        else:
            save_checkpoint(model, optimizer, epoch, val_loss, best=False)
            patience += 1
            save_log(f"[EarlyStopping] No improvement ({patience}/{PATIENCE})")
            if patience >= PATIENCE:
                save_log("[STOP] Early stopping triggered.")
                break

    save_log("[DONE] Training complete.")


if __name__ == "__main__":
    main()
