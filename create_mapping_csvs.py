import os
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path.cwd()

# Candidate locations for metadata and features (checked in order)
META_CANDIDATES = [
    PROJECT_ROOT / "metadata",
    PROJECT_ROOT / "data" / "SEP28K" / "metadata",
    PROJECT_ROOT / "data" / "SEP28K",  # in case metadata csvs are directly under SEP28K
]

FEAT_CANDIDATES = [
    PROJECT_ROOT / "features",
    PROJECT_ROOT / "data" / "SEP28K" / "features",
]

SPLITS = {
    "train": {"meta_name": "train.csv", "feat_subdir": "train", "out": PROJECT_ROOT / "features" / "mapping_train.csv"},
    "val":   {"meta_name": "val.csv",   "feat_subdir": "val",   "out": PROJECT_ROOT / "features" / "mapping_val.csv"},
    "test":  {"meta_name": "test.csv",  "feat_subdir": "test",  "out": PROJECT_ROOT / "features" / "mapping_test.csv"},
}

# Heuristic to detect clip column in metadata
def detect_clip_column(df):
    common = ['rel_path','path','wav_path','wav','file','filename','clip','clip_id','clipid','ClipId','clipid','keycode','audio']
    for c in df.columns:
        if c.lower() in common:
            return c
    # fallback: search any column whose values look like filenames or contain ".wav" or "SEP-" prefix
    for c in df.columns:
        try:
            sample = str(df[c].dropna().iloc[0])
        except Exception:
            sample = ""
        s = sample.lower()
        if ".wav" in s or s.startswith("sep-") or (s.count("_")>1 and any(ch.isdigit() for ch in s)):
            return c
    # else use first column
    return df.columns[0] if len(df.columns) else None

def normalize_basename(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    # if URL, take last path component
    if s.startswith("http://") or s.startswith("https://"):
        s = s.split("/")[-1]
    s = s.split("?")[0]
    basename = os.path.basename(s)
    stem = os.path.splitext(basename)[0]
    # remove trailing / leading whitespace and common prefixes
    stem = stem.strip()
    return stem

def find_existing_dir(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

def build_mapping_for_split(split_name, meta_path, feat_dir, out_csv, missing_list):
    print(f"\n== Processing split: {split_name} ==")
    if not meta_path.exists():
        print(f"[SKIP] metadata file not found: {meta_path}")
        return 0,0
    if not feat_dir.exists():
        print(f"[WARN] feature dir not found: {feat_dir} (creating)")
        feat_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path)
    clip_col = detect_clip_column(df)
    if clip_col is None:
        print("[ERROR] no column found in metadata to identify clips.")
        return 0,0
    print(f"Using metadata column '{clip_col}' to map features (metadata columns: {list(df.columns)})")

    rows = []
    missing = []
    found = 0

    # Build lookup map of available features (stem -> path)
    available = {}
    for p in feat_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".npy", ".npz"]:
            available[p.stem] = p

    for idx, row in df.iterrows():
        raw = row.get(clip_col, "")
        stem = normalize_basename(raw)
        if not stem:
            missing.append((idx, raw))
            continue
        # try direct stem match
        matched_path = available.get(stem)
        # try case-insensitive matches
        if not matched_path:
            for s,pth in available.items():
                if s.lower() == stem.lower():
                    matched_path = pth
                    break
        # try with underscores/spaces normalized
        if not matched_path:
            norm = stem.replace(" ", "_")
            for s,pth in available.items():
                if s.replace(" ", "_").lower() == norm.lower():
                    matched_path = pth
                    break
        # fallback: glob search for stem anywhere in filename
        if not matched_path:
            candidates = list(feat_dir.rglob(f"*{stem}*"))
            for c in candidates:
                if c.suffix.lower() in [".npy", ".npz"]:
                    matched_path = c
                    break

        if matched_path:
            rel = os.path.relpath(matched_path, PROJECT_ROOT)
            # try to extract label(s) - be permissive
            # prefer explicit label columns if present
            label_val = None
            for c in ['label','class','target','event','ground_truth']:
                if c in df.columns:
                    label_val = row[c]
                    break
            # if no single label column, try constructing a tuple of stutter counts if those columns exist
            if label_val is None:
                label_cols = [c for c in df.columns if any(k in c.lower() for k in ['block','prolong','soundrep','wordrep','interject','no_stutter','no stutter','no-stutter'])]
                if label_cols:
                    # create dict of values
                    label_val = {c: int(row[c]) if pd.notna(row[c]) and str(row[c]).isdigit() else row[c] for c in label_cols}
                else:
                    label_val = None
            rows.append({"feature_path": rel, "label": label_val})
            found += 1
        else:
            missing.append((idx, raw))

    # write mapping CSV (ensure parent dir exists)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        print("[WARN] No mappings found for this split - output will be empty.")
        df_out.to_csv(out_csv, index=False)
    else:
        df_out.to_csv(out_csv, index=False)
    print(f"Wrote mapping -> {out_csv}  (found: {found}, missing: {len(missing)})")

    for idx, raw in missing:
        missing_list.append(f"{split_name},{idx},{raw}")
    return found, len(missing)

def main():
    meta_dir = find_existing_dir(META_CANDIDATES)
    feat_base = find_existing_dir(FEAT_CANDIDATES)

    if meta_dir is None:
        print("ERROR: Could not find metadata directory. Checked candidates:")
        for p in META_CANDIDATES:
            print("  -", p)
        sys.exit(1)
    print("Using metadata directory:", meta_dir)

    if feat_base is None:
        # still allow features to be created under project root /features
        fallback = PROJECT_ROOT / "features"
        print("Feature directory not found in expected places. Will use/create:", fallback)
        feat_base = fallback

    print("Using features base directory:", feat_base)

    total_found = 0
    total_missing = 0
    missing_all = []

    for split, info in SPLITS.items():
        meta_path = meta_dir / info['meta_name']
        feat_dir = feat_base / info['feat_subdir']
        out_csv = PROJECT_ROOT / "features" / f"mapping_{split}.csv"  # keep mapping files at project_root/features/
        f, m = build_mapping_for_split(split, meta_path, feat_dir, out_csv, missing_all)
        total_found += f
        total_missing += m

    # write missing features file into metadata dir
    out_missing = meta_dir / "missing_features.txt"
    if missing_all:
        with out_missing.open("w", encoding="utf-8") as f:
            for line in missing_all:
                f.write(line + "\n")
        print(f"\nMissing mapping entries written to: {out_missing} (count: {len(missing_all)})")
    else:
        if out_missing.exists():
            out_missing.unlink()
        print("\nNo missing feature files detected.")

    print(f"\nSummary: total found features: {total_found}  total missing: {total_missing}")

if __name__ == "__main__":
    main()