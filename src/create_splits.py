import os
import sys
import re
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path.cwd()
DATA_DIR = BASE / "data" / "SEP28K"
CLIPS_DIR = DATA_DIR / "clips"
LABELS_CSV = DATA_DIR / "SEP-28k_labels.csv"
OUT_METADATA_DIR = BASE / "metadata"
OUT_METADATA_DIR.mkdir(exist_ok=True)

print(f"[INFO] Project base: {BASE}")
print(f"[INFO] Labels CSV: {LABELS_CSV}")
print(f"[INFO] Clips dir: {CLIPS_DIR}")
if not LABELS_CSV.exists():
    print("[ERROR] Labels CSV not found at:", LABELS_CSV)
    sys.exit(1)
if not CLIPS_DIR.exists():
    print("[ERROR] Clips dir not found at:", CLIPS_DIR)
    sys.exit(1)

# Load labels
labels = pd.read_csv(LABELS_CSV)
print("[INFO] Loaded labels:", labels.shape)

# Verify required columns
required = ['EpId','ClipId','Start','Stop','Prolongation','Block','SoundRep','WordRep','Interjection','NoStutteredWords']
missing_cols = [c for c in required if c not in labels.columns]
if missing_cols:
    print("[ERROR] Missing required columns in labels CSV:", missing_cols)
    sys.exit(1)

# Build file stem -> full path map (first occurrence)
audio_exts = {'.wav', '.mp3', '.flac', '.WAV', '.MP3', '.FLAC', '.m4a', '.M4A'}
file_map = {}
total_files = 0
for root, _, files in os.walk(CLIPS_DIR):
    for fn in files:
        total_files += 1
        stem = os.path.splitext(fn)[0]
        ext = os.path.splitext(fn)[1]
        if ext in audio_exts:
            if stem not in file_map:
                file_map[stem] = Path(root) / fn

print(f"[INFO] Scanned clips directory. Found {total_files} files, {len(file_map)} unique stems with audio extensions.")

# Precompute digits/groups in stems for fuzzy matching
digit_re = re.compile(r'(\d{1,8})')
stem_digits = {stem: digit_re.findall(stem) for stem in file_map.keys()}

# Matching strategies counters
strategy_counts = {"exact":0, "epid_clipid":0, "suffix_underscore":0, "separator_substr":0, "digits_in_stem":0}
matched = 0
missing_clipids = []
rows = []

for idx, r in labels.iterrows():
    clipid = str(r['ClipId'])
    epid = str(int(r['EpId'])) if pd.notnull(r['EpId']) else ""
    found_path = None
    strat = None

    # 1) exact stem
    if clipid in file_map:
        found_path = file_map[clipid]
        strat = "exact"
        strategy_counts['exact'] += 1

    # 2) epid_clipid
    if found_path is None:
        cand = f"{epid}_{clipid}"
        if cand in file_map:
            found_path = file_map[cand]
            strat = "epid_clipid"
            strategy_counts['epid_clipid'] += 1

    # 3) suffix underscore (e.g., FluencyBank_010_0)
    if found_path is None:
        suffix = f"_{clipid}"
        for stem, path in file_map.items():
            if stem.endswith(suffix):
                found_path = path
                strat = "suffix_underscore"
                strategy_counts['suffix_underscore'] += 1
                break

    # 4) separator substring patterns
    if found_path is None:
        patterns = [f"_{clipid}_", f"-{clipid}-", f".{clipid}."]
        for stem, path in file_map.items():
            if any(p in stem for p in patterns):
                found_path = path
                strat = "separator_substr"
                strategy_counts['separator_substr'] += 1
                break

    # 5) digits in stem
    if found_path is None:
        for stem, digits in stem_digits.items():
            if clipid in digits:
                found_path = file_map[stem]
                strat = "digits_in_stem"
                strategy_counts['digits_in_stem'] += 1
                break

    if found_path is None:
        rel_path = ""
        missing_clipids.append(clipid)
    else:
        rel_path = str(found_path.relative_to(BASE))
        matched += 1

    # duration computation (Start/Stop -> duration samples -> seconds; best-effort heuristic)
    try:
        start = float(r['Start']); stop = float(r['Stop'])
        dur_samples = max(0.0, stop - start)
    except Exception:
        dur_samples = np.nan

    dur_seconds = np.nan
    if pd.notnull(dur_samples):
        # heuristic conversions
        if dur_samples > 1e6:  # large -> samples at 16k
            dur_seconds = dur_samples / 16000.0
        elif dur_samples > 1000:
            # prefer samples@16k if reasonable
            if (dur_samples / 16000.0) <= 10.0:
                dur_seconds = dur_samples / 16000.0
            else:
                dur_seconds = dur_samples / 1000.0
        else:
            dur_seconds = dur_samples  # small => already seconds

    rows.append({
        "ClipId": clipid,
        "EpId": int(r['EpId']),
        "rel_path": rel_path,
        "duration_samples": dur_samples,
        "duration_seconds": dur_seconds,
        "match_strategy": strat or "",
        "label_block": int(r['Block']),
        "label_prolong": int(r['Prolongation']),
        "label_soundrep": int(r['SoundRep']),
        "label_wordrep": int(r['WordRep']),
        "label_interjection": int(r['Interjection']),
        "label_no_stutter": int(r['NoStutteredWords'])
    })

# Write master CSV
master = pd.DataFrame(rows)
master_fp = OUT_METADATA_DIR / "master_raw.csv"
master.to_csv(master_fp, index=False)
print(f"[INFO] Wrote master metadata -> {master_fp}")

# Missing files
missing_fp = OUT_METADATA_DIR / "missing_files.txt"
with open(missing_fp, "w") as f:
    for m in missing_clipids:
        f.write(m + "\n")
print(f"[INFO] Missing audio files count: {len(missing_clipids)} -> {missing_fp}")

print(f"[INFO] Matched {matched} / {len(labels)} label rows to audio files.")
print("[INFO] Match strategy counts:", strategy_counts)

# Show small samples
print("\n[SAMPLE matched rows (first 8 where rel_path != '')]:")
print(master[master['rel_path'] != ""].head(8).to_string(index=False))

print("\n[SAMPLE unmatched ClipIds (first 20)]:")
print(missing_clipids[:20])

# Proceed to multilabel stratified split only if there are matched rows
valid = master[master['rel_path'] != ""].reset_index(drop=True)
if valid.shape[0] == 0:
    print("[ERROR] No matched audio rows. Fix naming or provide audio files. Exiting.")
    sys.exit(1)

# Prepare for multilabel split
label_cols = ['label_block','label_prolong','label_soundrep','label_wordrep','label_interjection','label_no_stutter']
Y = valid[label_cols].values

# Import iterative stratifier
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:
    print("[ERROR] iterative-stratification not installed. Install with:")
    print("    pip install iterative-stratification")
    raise

# Create n folds and select folds for approx 70/15/15
n_splits = 20
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(mskf.split(valid, Y))
print(f"[INFO] Created {n_splits} stratified folds.")

# Fold assignment: 14 train (~70%), 3 val (~15%), 3 test (~15%)
train_folds = set(range(0,14))
val_folds = set(range(14,17))
test_folds = set(range(17,20))

train_idx = []
val_idx = []
test_idx = []

for i, (_, idxs) in enumerate(folds):
    if i in train_folds:
        train_idx += list(idxs)
    elif i in val_folds:
        val_idx += list(idxs)
    elif i in test_folds:
        test_idx += list(idxs)

train_df = valid.iloc[sorted(train_idx)].reset_index(drop=True)
val_df = valid.iloc[sorted(val_idx)].reset_index(drop=True)
test_df = valid.iloc[sorted(test_idx)].reset_index(drop=True)

print(f"[INFO] Final split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# Write splits
train_df.to_csv(OUT_METADATA_DIR / "train.csv", index=False)
val_df.to_csv(OUT_METADATA_DIR / "val.csv", index=False)
test_df.to_csv(OUT_METADATA_DIR / "test.csv", index=False)
print(f"[DONE] Saved train/val/test CSVs to {OUT_METADATA_DIR}")