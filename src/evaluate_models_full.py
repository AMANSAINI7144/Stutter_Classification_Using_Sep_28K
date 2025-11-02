import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, roc_auc_score,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Adjust these if your file names differ ----
PROJECT_ROOT = Path.cwd()
FEATURES_MAPPING = PROJECT_ROOT / "features" / "mapping_test.csv"
FEATURES_BASE = PROJECT_ROOT  # feature paths in mapping are relative to project root
CHECKPOINT_HYBRID = PROJECT_ROOT / "models" / "checkpoints" / "best_model_hybrid.pth"
CHECKPOINT_ATT = PROJECT_ROOT / "models" / "checkpoints_attention_models" / "best_model_attention.pth"

# Output base
OUT_BASE = PROJECT_ROOT / "results" / "evaluation"

# model import paths - these refer to src/models/*.py
from models.crnn_model import CRNN_GRU
from models.crnn_attention import CRNN_Attention

LABEL_NAMES = ['block', 'prolong', 'soundrep', 'wordrep', 'interjection', 'no_stutter']

# ---------------- Dataset ----------------
import ast
import json

class EvalFeatureDataset(Dataset):
    def __init__(self, mapping_csv):
        self.df = pd.read_csv(mapping_csv)
        # paths
        if 'feature_path' not in self.df.columns:
            # maybe the first column is feature path under different name
            # try to locate any column that looks like a relative path to .npy
            cand = None
            for c in self.df.columns:
                sample = str(self.df[c].dropna().astype(str).iloc[0]) if len(self.df) else ""
                if sample.endswith(".npy") or sample.endswith(".npz"):
                    cand = c
                    break
            if cand is None:
                raise ValueError("mapping CSV must contain a 'feature_path' column or a column with .npy paths.")
            else:
                self.df = self.df.rename(columns={cand: 'feature_path'})

        # If explicit six label columns exist, use them directly
        possible = ['label_block','label_prolong','label_soundrep','label_wordrep','label_interjection','label_no_stutter']
        if all(c in self.df.columns for c in possible):
            self.labels = self.df[possible].values.astype(float)
        else:
            # try a single 'label' column present (common)
            if 'label' in self.df.columns:
                # parse each label entry into a vector of length 6 (in LABEL_NAMES order)
                parsed_labels = []
                for val in self.df['label'].tolist():
                    if pd.isna(val):
                        parsed_labels.append([0.0]*6)
                        continue
                    # if already a dict (pandas might have preserved it)
                    if isinstance(val, dict):
                        d = val
                    else:
                        s = str(val).strip()
                        d = None
                        # try JSON
                        try:
                            d = json.loads(s)
                        except Exception:
                            # try Python literal (e.g. "{'label_block':1, ...}")
                            try:
                                d = ast.literal_eval(s)
                            except Exception:
                                d = None
                    if isinstance(d, dict):
                        # map label names to numbers, default 0
                        row_vals = []
                        # try to match keys flexibly (allow keys like 'block' or 'label_block')
                        for label in LABEL_NAMES:
                            # exact match
                            if label in d:
                                v = d[label]
                            else:
                                # try variations: remove 'label_' prefix
                                short = label.replace('label_', '')
                                # find first key containing the short name
                                found_key = None
                                for k in d.keys():
                                    if short in k.lower():
                                        found_key = k
                                        break
                                if found_key is not None:
                                    v = d[found_key]
                                else:
                                    v = 0
                            try:
                                v = float(v)
                            except Exception:
                                # non-numeric -> try to convert booleans or strings
                                if isinstance(v, bool):
                                    v = 1.0 if v else 0.0
                                else:
                                    try:
                                        v = float(str(v))
                                    except Exception:
                                        v = 0.0
                            row_vals.append(v)
                        parsed_labels.append(row_vals)
                    else:
                        # Could not parse to dict: handle simple numeric or categorical label
                        # If it's a single number (0..1 or 0..3) -> assume it's aggregated "no_stutter" or "count" for first label?
                        # Safer default: treat scalar as single-label index if integer and <=5
                        try:
                            num = float(s)
                            # if integer in 0..5 interpret as index -> one-hot
                            if abs(num - round(num)) < 1e-6 and 0 <= int(round(num)) < 6:
                                idx = int(round(num))
                                onehot = [0.0]*6
                                onehot[idx] = 1.0
                                parsed_labels.append(onehot)
                            else:
                                # otherwise treat as probability for 'no_stutter' as fallback
                                parsed_labels.append([0.0,0.0,0.0,0.0,0.0, float(num)])
                        except Exception:
                            # last fallback: zeros
                            parsed_labels.append([0.0]*6)
                self.labels = np.array(parsed_labels, dtype=float)
            else:
                # no label columns found at all
                raise ValueError("Could not find required 6 label columns or a single 'label' column in mapping CSV. Found: " + str(self.df.columns.tolist()))

        # store feature paths
        self.feature_paths = [Path(PROJECT_ROOT) / p for p in self.df['feature_path'].tolist()]

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        fp = self.feature_paths[idx]
        x = np.load(fp)
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y, str(fp)

# ---------------- Utilities ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def plot_and_save_confmat(y_true, y_pred, label, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {label}')
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_and_save_roc(y_true, y_score, label, outpath):
    # y_true: binary, y_score: probabilities
    try:
        # compute ROC AUC and ROC curve points using sklearn
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC - {label}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return roc_auc
    except Exception as e:
        return None

# ---------------- Evaluation core ----------------
def run_inference_and_metrics(model, ckpt_path, model_name, device):
    # Prepare output folders
    outdir = OUT_BASE / model_name
    tables_dir = outdir / "tables"
    plots_dir = outdir / "plots"
    ensure_dir(outdir); ensure_dir(tables_dir); ensure_dir(plots_dir)

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Robustly find where the model weights live in the checkpoint
    state = None
    if isinstance(ckpt, dict):
        # Common keys in different training scripts: try several names
        for key in ['model_state_dict', 'state_dict', 'model_state', 'model_state_dicts', 'model_state_dict_']:
            if key in ckpt:
                state = ckpt[key]
                print(f"[INFO] Using checkpoint key: '{key}' for model weights.")
                break
        # some scripts store under 'model' or 'net'
        if state is None:
            for key in ['model', 'net']:
                if key in ckpt and isinstance(ckpt[key], dict):
                    state = ckpt[key]
                    print(f"[INFO] Using checkpoint key: '{key}' for model weights.")
                    break
    # If still None and ckpt looks like a state dict (keys look like 'cnn.0.weight'), use it
    if state is None:
        # check first key to guess whether ckpt is a state_dict
        try:
            first_key = next(iter(ckpt.keys()))
            if isinstance(first_key, str) and ('.' in first_key):
                state = ckpt
                print("[INFO] Checkpoint appears to be a raw state_dict (no wrapper).")
        except Exception:
            pass

    if state is None:
        raise RuntimeError(f"Could not find model weights inside checkpoint {ckpt_path}. Available keys: {list(ckpt.keys())}")

    # load into model
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    # dataset
    ds = EvalFeatureDataset(FEATURES_MAPPING)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    all_probs = []
    all_preds = []
    all_targets = []
    paths = []

    with torch.no_grad():
        for X, y, pths in loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            # handle attention model returning (out, attn)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            # if model returns logits (CRNN_GRU) we need sigmoid
            out = torch.sigmoid(out)
            probs = out.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())
            paths.extend(list(pths))

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Normalize/threshold targets: training used threshold 0.33 for annotator counts
    bin_targets = (all_targets > 0.33).astype(int)

    # Save predictions table
    n = all_probs.shape[0]
    rows = []
    for i in range(n):
        row = {"feature_path": paths[i]}
        for j,label in enumerate(LABEL_NAMES):
            row[f"prob_{label}"] = float(all_probs[i,j])
            row[f"pred_{label}"] = int(all_preds[i,j])
            row[f"true_{label}"] = int(bin_targets[i,j])
        rows.append(row)
    df_preds = pd.DataFrame(rows)
    df_preds.to_csv(tables_dir / "predictions.csv", index=False)

    # Compute per-label metrics
    per_label = []
    for j,label in enumerate(LABEL_NAMES):
        y_true = bin_targets[:,j]
        y_pred = all_preds[:,j]
        # safety: if all y_true same class, some metrics undefined; handle gracefully
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        # roc auc if possible
        try:
            auc = roc_auc_score(y_true, all_probs[:,j])
        except Exception:
            auc = None
        per_label.append({
            "label": label, "precision": p, "recall": r, "f1": f1, "accuracy": acc, "roc_auc": auc,
            "support": int(y_true.sum()), "n_samples": len(y_true)
        })
        # save confusion matrix plot
        plot_and_save_confmat(y_true, y_pred, label, plots_dir / f"confmat_{label}.png")
        # save ROC plot if possible
        if auc is not None:
            plot_and_save_roc(y_true, all_probs[:,j], label, plots_dir / f"roc_{label}.png")

    df_per_label = pd.DataFrame(per_label)
    df_per_label.to_csv(tables_dir / "per_label_metrics.csv", index=False)

    # Overall metrics: macro-averaged
    macro_precision = precision_score(bin_targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(bin_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(bin_targets, all_preds, average='macro', zero_division=0)
    # subset accuracy (exact match)
    subset_acc = accuracy_score(bin_targets.tolist(), all_preds.tolist())

    report = classification_report(bin_targets, all_preds, target_names=LABEL_NAMES, zero_division=0)
    with open(tables_dir / "classification_report.txt", "w") as f:
        f.write(report)

    metrics = {
        "model": model_name,
        "n_samples": int(all_probs.shape[0]),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "subset_accuracy": float(subset_acc)
    }
    # save metrics json and table
    with open(tables_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    # also save summary CSV
    pd.DataFrame([metrics]).to_csv(tables_dir / "metrics_summary.csv", index=False)

    print(f"\nSaved evaluation outputs for {model_name} -> {outdir}")
    return metrics, df_per_label, df_preds

# ---------------- Main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dir(OUT_BASE)

    # --- Evaluate Hybrid CRNN ---
    print("\nEvaluating CRNN_Hybrid ...")
    model_h = CRNN_GRU(num_classes=len(LABEL_NAMES))
    if not CHECKPOINT_HYBRID.exists():
        print("WARNING: Hybrid checkpoint not found:", CHECKPOINT_HYBRID)
    else:
        run_inference_and_metrics(model_h, CHECKPOINT_HYBRID, "CRNN_Hybrid", device)

    # --- Evaluate Attention CRNN ---
    print("\nEvaluating CRNN_Attention ...")
    model_a = CRNN_Attention(num_classes=len(LABEL_NAMES))
    if not CHECKPOINT_ATT.exists():
        print("WARNING: Attention checkpoint not found:", CHECKPOINT_ATT)
    else:
        run_inference_and_metrics(model_a, CHECKPOINT_ATT, "CRNN_Attention", device)

    print("\nAll evaluations complete. Results saved under:", OUT_BASE)

if __name__ == "__main__":
    main()