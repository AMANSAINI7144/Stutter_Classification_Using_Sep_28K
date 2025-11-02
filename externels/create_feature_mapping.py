#!/usr/bin/env python3
"""
Create mapping_<split>.csv files in features/<split>/ from metadata/<split>.csv
and the files actually present in features/<split>/.

Outputs:
  features/<split>/mapping_<split>.csv
Each row includes feature_path (absolute), rel_path (if available in metadata),
ClipId, EpId, and the 6 label columns from metadata.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "features")
METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
SPLITS = ["train", "val", "test"]

LABEL_COLS = [
    "label_block",
    "label_prolong",
    "label_soundrep",
    "label_wordrep",
    "label_interjection",
    "label_no_stutter",
]

# Helper to find .npy files in a directory and return a set of stems (no ext)
def collect_feature_stems(split_dir):
    stems = {}
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(".npy"):
                stem = os.path.splitext(f)[0]
                path = os.path.join(root, f)
                stems[stem] = path
    return stems

for split in SPLITS:
    print(f"[INFO] Processing split: {split}")
    meta_path = os.path.join(METADATA_DIR, f"{split}.csv")
    if not os.path.exists(meta_path):
        print(f"[WARN] metadata/{split}.csv not found — skipping {split}")
        continue

    df_meta = pd.read_csv(meta_path)
    # normalize column names if needed
    # expected columns in metadata: ClipId, EpId, rel_path (maybe), and label columns
    # if metadata stores label counts as ints, keep them here; dataset will normalize.
    feature_split_dir = os.path.join(FEATURE_DIR, split)
    if not os.path.isdir(feature_split_dir):
        print(f"[WARN] features/{split}/ not found — skipping {split}")
        continue

    feature_stems = collect_feature_stems(feature_split_dir)
    print(f"[INFO] Found {len(feature_stems)} .npy feature files under features/{split}")

    rows = []
    missing_features = []
    for idx, row in df_meta.iterrows():
        # Try to infer feature file stem names:
        # Common conventions:
        #  - maybe feature files are saved using ClipId or some clip identifier.
        # We'll try the obvious candidates:
        cand_stems = []

        # if metadata has a 'feature_stem' column (rare), use it
        if "feature_stem" in df_meta.columns:
            cand_stems.append(str(row["feature_stem"]))
        # if feature files were saved as ClipId
        if "ClipId" in df_meta.columns:
            cand_stems.append(str(row["ClipId"]))
        # if metadata has a 'rel_path' or 'path' column, often the feature stem is basename without ext
        if "rel_path" in df_meta.columns and pd.notna(row["rel_path"]):
            cand_stems.append(os.path.splitext(os.path.basename(row["rel_path"]))[0])
        if "path" in df_meta.columns and pd.notna(row["path"]):
            cand_stems.append(os.path.splitext(os.path.basename(row["path"]))[0])

        # As a fallback try episode+clip like "HeStutters_0_9" if such columns exist
        if "EpId" in df_meta.columns and "ClipId" in df_meta.columns:
            cand_stems.append(f"{row.get('Show','')}_{int(row['EpId'])}_{int(row['ClipId'])}".strip("_"))

        # dedupe and try to match feature files
        matched = False
        for s in cand_stems:
            if s is None:
                continue
            s = str(s)
            if s in feature_stems:
                feat_path = feature_stems[s]
                matched = True
                break

        if matched:
            # collect labels (if present)
            label_vals = {}
            for c in LABEL_COLS:
                label_vals[c] = row[c] if c in df_meta.columns else 0
            rows.append({
                "ClipId": row.get("ClipId", ""),
                "EpId": row.get("EpId", ""),
                "rel_path": row.get("rel_path", ""),
                "feature_path": feat_path,
                **label_vals
            })
        else:
            missing_features.append(row.get("ClipId", row.to_dict()))

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(feature_split_dir, f"mapping_{split}.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote mapping {out_csv} with {len(out_df)} rows")
    if missing_features:
        miss_path = os.path.join(feature_split_dir, f"missing_features_{split}.csv")
        pd.DataFrame(missing_features).to_csv(miss_path, index=False)
        print(f"[WARN] {len(missing_features)} metadata rows had no matching .npy — see {miss_path}")
    print()
