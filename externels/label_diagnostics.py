import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def to_binary_series(s):
    # convert common truthy strings to 1, everything else to 0
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).clip(0,1)
    s = s.fillna("").astype(str).str.strip()
    trues = set(["1", "true", "True", "TRUE", "yes", "Yes", "Y", "y"])
    return s.apply(lambda x: 1 if x in trues else 0)

def main():
    p = Path(args.mapping)
    if not p.exists():
        print("ERROR: mapping file not found:", p)
        return

    df = pd.read_csv(p)
    # Select label columns: those starting with label_ (case-insensitive)
    label_cols = [c for c in df.columns if c.lower().startswith("label_")]
    if not label_cols:
        print("No columns starting with 'label_' found in", p)
        print("Available columns:", list(df.columns))
        return

    out_rows = []
    pos_weights = []
    print("Detected label columns:", label_cols)
    for c in label_cols:
        s_raw = df[c]
        s = to_binary_series(s_raw)
        pos = int((s == 1).sum())
        neg = int((s == 0).sum())
        frac = pos / (pos + neg) if (pos + neg) > 0 else 0.0
        # safe pos_weight: neg/pos (avoid division by zero)
        if pos == 0:
            pw = float(neg)
        else:
            pw = float(neg) / float(pos)
        pos_weights.append(pw)
        out_rows.append({"label": c, "pos": pos, "neg": neg, "pos_frac": round(frac, 6), "pos_weight": round(pw, 6)})

    out_df = pd.DataFrame(out_rows).set_index("label")
    print("\nPer-label counts and suggested pos_weight (neg/pos):")
    print(out_df.to_string())
    # ensure results dir exists
    Path("results/experiments").mkdir(parents=True, exist_ok=True)
    out_df.to_csv("results/experiments/label_diagnostics.csv")
    print("\nSaved results/experiments/label_diagnostics.csv")
    print("\nSuggested pos_weight vector (rounded):")
    print([round(x,3) for x in pos_weights])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=str, default="features/train/mapping_train.csv",
                        help="path to mapping CSV with label_ columns")
    args = parser.parse_args()
    main()