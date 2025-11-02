import argparse, pandas as pd, numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score

def main(args):
    preds = pd.read_csv(args.preds)
    thr_df = pd.read_csv(args.thr, index_col=0)
    # thr_df index should be label names (matching prob column suffix)
    # find probability columns
    prob_cols = [c for c in preds.columns if c.startswith(args.prob_prefix)]
    # derive label names from prob cols
    labels = [c[len(args.prob_prefix):] if c.startswith(args.prob_prefix) else c for c in prob_cols]
    # build threshold array in same order
    thresholds = []
    for lab in labels:
        if lab in thr_df.index:
            thresholds.append(float(thr_df.loc[lab, "best_threshold"]))
        else:
            # try variants
            for cand in [lab, "label_"+lab, "true_"+lab]:
                if cand in thr_df.index:
                    thresholds.append(float(thr_df.loc[cand, "best_threshold"]))
                    break
            else:
                thresholds.append(0.5)
    thresholds = np.array(thresholds)

    probs = preds[prob_cols].values
    y_true_cols = []
    for lab in labels:
        # try several names for true column
        possible = [f"true_{lab}", lab, f"label_{lab}"]
        found = None
        for p in possible:
            if p in preds.columns:
                found = p; break
        if found is None:
            # fallback zeros
            y_true_cols.append(np.zeros(len(preds), dtype=int))
        else:
            y_true_cols.append(preds[found].astype(int).values)
    y_true = np.stack(y_true_cols, axis=1)

    y_pred = (probs >= thresholds[None, :]).astype(int)

    # Save tuned predictions
    out_df = preds.copy()
    for i, c in enumerate(prob_cols):
        out_df["tuned_"+c] = y_pred[:, i]
    out_csv = args.preds.replace(".csv", "_tuned.csv")
    out_df.to_csv(out_csv, index=False)
    print("Saved tuned predictions to", out_csv)

    # metrics
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    print("Tuned threshold macro F1:", macro, "micro F1:", micro)

    # pretty classification report
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    with open(args.out, "w") as f:
        f.write(report)
    print("Saved classification report to", args.out)
    # per-label CSV
    p, r, f1s, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    per_df = pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f1s, "support": sup})
    per_df.to_csv(args.out.replace(".txt", "_per_label.csv"), index=False)
    print("Saved per-label CSV to", args.out.replace(".txt", "_per_label.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--thr", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--prob_prefix", type=str, default="prob_")
    args = parser.parse_args()
    main(args)