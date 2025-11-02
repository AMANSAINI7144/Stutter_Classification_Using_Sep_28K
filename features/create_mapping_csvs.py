import os
import csv
import pandas as pd
from pathlib import Path

PROJECT_ROOT = os.getcwd()   # run script from project root (~/projects/MPC_Project)
BASE = Path(PROJECT_ROOT) / "data" / "SEP28K"
META_DIR = BASE / "metadata"
FEAT_DIR = BASE / "features"
OUT_MISSING = META_DIR / "missing_features.txt"

SPLITS = {
    "train": {"meta": META_DIR / "train.csv", "feat_dir": FEAT_DIR / "train", "out": FEAT_DIR / "mapping_train.csv"},
    "val":   {"meta": META_DIR / "val.csv",   "feat_dir": FEAT_DIR / "val",   "out": FEAT_DIR / "mapping_val.csv"},
    "test":  {"meta": META_DIR / "test.csv",  "feat_dir": FEAT_DIR / "test",  "out": FEAT_DIR / "mapping_test.csv"},
}

# Helper: find column name that likely contains clip identifier / wav path
def detect_clip_column(df):
    # common names
    common = ['wav_path','wav','file','filename','clip','clip_id','clipname','keycode','path','audio']
    for c in common:
        if c in df.columns:
            return c
    # fallback: search any column whose values look like filenames or contain ".wav" or "SEP-" prefix
    for c in df.columns:
        sample = str(df[c].iloc[0]) if len(df) else ""
        if isinstance(sample, str):
            if ".wav" in sample.lower() or sample.startswith("SEP-") or sample.count("_")>1:
                return c
    # final fallback: use first column
    return df.columns[0] if len(df.columns) else None

def normalize_basename(x):
    # x may be a path, url, numeric id, etc. Return basename without extension
    if pd.isna(x):
        return None
    s = str(x).strip()
    # if URL, take last path component
    if s.startswith("http://") or s.startswith("https://"):
        s = s.split("/")[-1]
    # remove query strings if any
    s = s.split("?")[0]
    # if contains timestamp or start/end, user may have provided something else; still take basename
    basename = os.path.basename(s)
    # remove extension
    stem = os.path.splitext(basename)[0]
    return stem

def build_mapping_for_split(split_name, meta_path, feat_dir, out_csv, missing_list):
    print(f"\n== Processing split: {split_name} ==")
    if not meta_path.exists():
        print(f"[SKIP] metadata file not found: {meta_path}")
        return 0,0
    if not feat_dir.exists():
        print(f"[WARN] feature dir not found (will create): {feat_dir}")
        feat_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path)
    clip_col = detect_clip_column(df)
    if clip_col is None:
        print("[ERROR] no column found in metadata to identify clips.")
        return 0,0
    print(f"Using metadata column '{clip_col}' to map features (columns: {list(df.columns)})")

    rows = []
    missing = []
    found = 0

    # Build quick lookup set of available feature basenames (without .npy)
    available_files = {p.stem for p in feat_dir.glob("*.npy")}
    # also consider files with uppercase or .npz etc
    available_files |= {p.stem for p in feat_dir.glob("*.npz")}
    # if features saved in nested dirs (like features/train/subdir/*.npy) include those
    for p in feat_dir.rglob("*.npy"):
        available_files.add(p.stem)

    for idx, row in df.iterrows():
        raw = row.get(clip_col, "")
        stem = normalize_basename(raw)
        if not stem:
            # maybe metadata has separate 'episode' + start/end to form unique name; skip gracefully
            missing.append((idx, raw))
            continue
        # possible feature filename candidates
        candidates = [
            f"{stem}.npy",
            f"{stem}.npz",
            f"{stem.lower()}.npy",
            f"{stem.upper()}.npy",
            f"{stem}.npy".replace(" ", "_"),
        ]
        matched_path = None
        for cand in candidates:
            fp = feat_dir / cand
            if fp.exists():
                matched_path = fp
                break
        # fallback: if available_files contains stem, find actual file path
        if not matched_path and stem in available_files:
            # find first matching path
            matches = list(feat_dir.rglob(f"{stem}.*"))
            if matches:
                matched_path = matches[0]

        if matched_path:
            # put relative path from project root; you can change to absolute if required
            rel = os.path.relpath(matched_path, PROJECT_ROOT)
            # try to get label from common columns
            label = None
            for c in ['label','class','target','event','ground_truth']:
                if c in df.columns:
                    label = row[c]
                    break
            # if no single label column, try 'labels' or count columns
            if label is None:
                # attempt to find any column whose name contains 'prolong' etc and build a majority label if present
                for k in ['prolong','block','sound','word','interject','no_stutter','no-stutter','no stutter']:
                    for col in df.columns:
                        if k in col.lower():
                            # if this is a count column, we'll lazy-choose highest count later; but for mapping keep single column if exists
                            pass
                # default fallback
                label = row.get('label', None)
            rows.append({"feature_path": rel, "label": label})
            found += 1
        else:
            missing.append((idx, raw))

    # write mapping CSV
    out_hdr = ['feature_path','label']
    out_path = out_csv
    pd.DataFrame(rows).to_csv(out_path, index=False, columns=out_hdr)
    print(f"Wrote mapping -> {out_path}  (found: {found}, missing: {len(missing)})")

    # append missing entries to missing_list for the whole dataset log
    for idx, raw in missing:
        missing_list.append(f"{split_name},{idx},{raw}")
    return found, len(missing)

def main():
    total_found = 0
    total_missing = 0
    missing_all = []

    for split, info in SPLITS.items():
        found, missing = build_mapping_for_split(split,
                                                info['meta'],
                                                Path(info['feat_dir']),
                                                Path(info['out']),
                                                missing_all)
        total_found += found
        total_missing += missing

    # save missing list
    if missing_all:
        OUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_MISSING, 'w', encoding='utf-8') as f:
            for line in missing_all:
                f.write(line + "\n")
        print(f"\nMissing mapping entries written to: {OUT_MISSING} (count: {len(missing_all)})")
    else:
        if OUT_MISSING.exists():
            OUT_MISSING.unlink()
        print("\nNo missing feature files detected.")

    print(f"\nSummary: total found features: {total_found}  total missing: {total_missing}")

if __name__ == "__main__":
    main()