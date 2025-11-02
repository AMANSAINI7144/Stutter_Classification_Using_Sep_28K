import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

def find_prob_true_cols(df, prob_prefix, true_prefix):
    prob_cols = [c for c in df.columns if c.startswith(prob_prefix)]
    true_cols = [c for c in df.columns if c.startswith(true_prefix)]
    # if explicit prefixes not found, try to infer by common patterns
    if not prob_cols:
        prob_cols = [c for c in df.columns if "prob" in c.lower() or "pred" in c.lower()]
    if not true_cols:
        # try to find columns that match label names without prob_ prefix
        # e.g., prob_label_block and label_block (no 'true_' prefix)
        inferred_true = []
        for p in prob_cols:
            base = p
            for prefix in ("prob_", "pred_", ""):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            # try exact match
            if base in df.columns:
                inferred_true.append(base)
            elif ("true_" + base) in df.columns:
                inferred_true.append("true_" + base)
        true_cols = list(dict.fromkeys(inferred_true))  # preserve order, unique
    # final sort to ensure same order
    if len(prob_cols) != len(true_cols):
        # try best-effort match by suffix
        matched_prob, matched_true = [], []
        for p in prob_cols:
            base = p
            for prefix in ("prob_", "pred_"):
                if base.startswith(prefix):
                    base = base[len(prefix):]
                    break
            candidates = [c for c in df.columns if c.endswith(base)]
            if candidates:
                matched_prob.append(p)
                matched_true.append(candidates[0])
        if matched_prob and len(matched_prob) == len(matched_true):
            prob_cols, true_cols = matched_prob, matched_true

    return prob_cols, true_cols

def sweep_best_thresholds(y_true, y_prob, min_thr=0.01, max_thr=0.99, step=0.01):
    thresholds = np.arange(min_thr, max_thr + 1e-9, step)
    best_thr = {}
    for j in range(y_true.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_prob[:, j] >= t).astype(int)
            f1 = f1_score(y_true[:, j], y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thr[j] = {"threshold": best_t, "f1": float(best_f1)}
    return best_thr

def main(args):
    preds_path = Path(args.preds)
    if not preds_path.exists():
        print("ERROR: predictions file not found:", preds_path)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)
    prob_cols, true_cols = find_prob_true_cols(df, args.prob_prefix, args.true_prefix)

    if not prob_cols:
        print("ERROR: could not detect probability columns. Please pass --prob_prefix or ensure file has 'prob_' columns.")
        print("Available columns:", list(df.columns)[:50])
        return
    if not true_cols or len(true_cols) != len(prob_cols):
        print("WARNING: true label columns not detected reliably; attempting to infer by prob column names.")
    print("Using prob columns:", prob_cols)
    print("Using true columns:", true_cols)

    # Build arrays
    y_prob = df[prob_cols].values
    # If true columns are not found, attempt to derive from prob cols by stripping 'prob_'
    if true_cols and len(true_cols) == len(prob_cols):
        y_true = df[true_cols].values.astype(int)
        label_names = [c.replace(args.prob_prefix, "") if c.startswith(args.prob_prefix) else c for c in prob_cols]
    else:
        # try to find corresponding true columns automatically
        label_names = []
        true_list = []
        for p in prob_cols:
            base = p
            if base.startswith(args.prob_prefix):
                base = base[len(args.prob_prefix):]
            # possible true column names
            candidates = [base, "true_" + base, "label_" + base, "y_" + base]
            found = None
            for cand in candidates:
                if cand in df.columns:
                    found = cand
                    break
            if found:
                true_list.append(found)
                label_names.append(base)
            else:
                # if none found, default to using prob column name as label name and assume no true column
                true_list.append(None)
                label_names.append(base)
        true_cols = true_list
        # Build y_true matrix, where missing true columns cause zeros
        y_true_cols = []
        for tc in true_cols:
            if tc is None:
                y_true_cols.append(np.zeros(len(df), dtype=int))
            else:
                y_true_cols.append(df[tc].astype(int).values)
        y_true = np.stack(y_true_cols, axis=1)

    # ensure shapes match
    if y_true.shape[1] != y_prob.shape[1]:
        print("ERROR: mismatch between number of true columns and prob columns:", y_true.shape, y_prob.shape)
        return

    # Sweep thresholds
    best_thr = sweep_best_thresholds(y_true, y_prob, min_thr=args.min_thr, max_thr=args.max_thr, step=args.step)

    # create results table
    rows = []
    for i, pcol in enumerate(prob_cols):
        label = label_names[i]
        thr = best_thr[i]["threshold"]
        f1 = best_thr[i]["f1"]
        rows.append({"label": label, "prob_col": pcol, "true_col": true_cols[i] if i < len(true_cols) else None,
                     "best_threshold": thr, "best_f1": f1})
    best_df = pd.DataFrame(rows).set_index("label")
    best_csv = out_dir / "best_thresholds.csv"
    best_df.to_csv(best_csv)
    print("Saved best thresholds to", best_csv)
    print(best_df)

    # Apply thresholds and compute classification report (before & after)
    # Before (default 0.5)
    y_pred_default = (y_prob >= 0.5).astype(int)
    macro_f1_default = f1_score(y_true, y_pred_default, average='macro', zero_division=0)
    micro_f1_default = f1_score(y_true, y_pred_default, average='micro', zero_division=0)
    print(f"Default thresholds (0.5) -> macro F1: {macro_f1_default:.4f}, micro F1: {micro_f1_default:.4f}")

    # After (per-label)
    thresholds_arr = np.array([best_thr[i]["threshold"] for i in range(y_prob.shape[1])])
    y_pred_tuned = (y_prob >= thresholds_arr[None, :]).astype(int)
    macro_f1_tuned = f1_score(y_true, y_pred_tuned, average='macro', zero_division=0)
    micro_f1_tuned = f1_score(y_true, y_pred_tuned, average='micro', zero_division=0)
    print(f"Tuned thresholds -> macro F1: {macro_f1_tuned:.4f}, micro F1: {micro_f1_tuned:.4f}")

    # Save tuned classification report
    label_display_names = [c for c in best_df.index.tolist()]
    try:
        report = classification_report(y_true, y_pred_tuned, target_names=label_display_names, zero_division=0)
    except Exception:
        report = classification_report(y_true, y_pred_tuned, zero_division=0)
    (out_dir / "tuned_classification_report.txt").write_text(report)
    print("Saved tuned classification report to", out_dir / "tuned_classification_report.txt")
    # Also save per-label metrics CSV
    p, r, f1s, sup = precision_recall_fscore_support(y_true, y_pred_tuned, zero_division=0)
    per_label_df = pd.DataFrame({"label": label_display_names, "precision": p, "recall": r, "f1": f1s, "support": sup})
    per_label_df.to_csv(out_dir / "tuned_per_label_metrics.csv", index=False)
    print("Saved tuned per-label metrics to", out_dir / "tuned_per_label_metrics.csv")

    print("\n=== Summary ===")
    print("Default (0.5) macro/micro F1:", round(macro_f1_default,4), round(micro_f1_default,4))
    print("Tuned thresholds macro/micro F1:", round(macro_f1_tuned,4), round(micro_f1_tuned,4))
    print("Best thresholds saved to:", best_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, default="results/evaluation/CRNN_Hybrid/predictions_val.csv", help="predictions csv")
    parser.add_argument("--prob_prefix", type=str, default="prob_", help="prefix for probability columns in preds csv")
    parser.add_argument("--true_prefix", type=str, default="true_", help="prefix for true label columns in preds csv")
    parser.add_argument("--out_dir", type=str, default="results/experiments", help="output directory")
    parser.add_argument("--min_thr", type=float, default=0.01)
    parser.add_argument("--max_thr", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()
    main(args)