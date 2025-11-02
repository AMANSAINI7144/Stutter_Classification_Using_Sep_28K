import os
from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# ---------------------------
# Config / Defaults
# ---------------------------
DEFAULT_SR = 16000          # target sample rate
N_MELS = 80
WIN_MS = 25                 # window length in ms
HOP_MS = 10                 # hop length in ms
POWER = 2.0                 # power spectrogram (mel spectrogram uses power)
EPS = 1e-6                  # numerical stability for log

# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_log_mel(y, sr, n_mels=N_MELS, win_ms=WIN_MS, hop_ms=HOP_MS, power=POWER):
    win_length = int(sr * win_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)
    # ensure FFT length at least win_length (choose next power of 2)
    n_fft = 1 << (win_length - 1).bit_length()
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, n_mels=n_mels, power=power)
    logS = librosa.power_to_db(S, ref=np.max)
    # convert to float32
    return logS.astype(np.float32), hop_length, win_length

def fixed_frames(logm, target_frames):
    """Pad or truncate spectrogram (n_mels, T) to (n_mels, target_frames)."""
    n_mels, T = logm.shape
    if T == target_frames:
        return logm
    if T < target_frames:
        pad_width = ((0,0), (0, target_frames - T))
        return np.pad(logm, pad_width, mode='constant', constant_values=logm.min())
    else:
        return logm[:, :target_frames]

def estimate_target_frames(sr, duration_sec=3.0, win_ms=WIN_MS, hop_ms=HOP_MS):
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if hop <= 0:
        hop = 1
    # frames = 1 + floor((samples - win)/hop)
    samples = int(sr * duration_sec)
    frames = 1 + int(math.floor(max(0, samples - win) / hop))
    return frames

# ---------------------------
# Main processing function
# ---------------------------
def process_split(csv_path: Path, out_dir: Path, sr=DEFAULT_SR, n_mels=N_MELS,
                  win_ms=WIN_MS, hop_ms=HOP_MS, duration_sec=3.0, force=False):
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"[INFO] Processing {csv_path.name} -> {total} samples")

    ensure_dir(out_dir)
    mapping_rows = []
    # Determine expected frame length (consistent across dataset)
    target_frames = estimate_target_frames(sr, duration_sec=duration_sec, win_ms=win_ms, hop_ms=hop_ms)
    print(f"[INFO] target sr={sr}, n_mels={n_mels}, win_ms={win_ms}, hop_ms={hop_ms}, target_frames={target_frames}")

    errors = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=total, desc=f"Extract {csv_path.name}")):
        try:
            clipid = str(getattr(row, "ClipId"))
            rel_path = getattr(row, "rel_path")
            if not rel_path or not Path(rel_path).exists():
                errors.append((clipid, rel_path, "missing_file"))
                continue

            # Read audio (soundfile preserves original samplerate)
            wav_path = str(Path(rel_path))
            y, orig_sr = sf.read(wav_path, always_2d=False)
            # convert multi-channel to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            # resample if needed
            if orig_sr != sr:
                y = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)

            # We expect clips to be ~duration_sec long (3s). If slightly different, we allow padding or trim.
            # Compute log-mel
            logm, hop_length, win_length = compute_log_mel(y, sr, n_mels=n_mels, win_ms=win_ms, hop_ms=hop_ms, power=POWER)

            # Fix frames to target_frames
            feat = fixed_frames(logm, target_frames)

            # Save .npy: features/<split>/<ClipId>.npy
            out_path = out_dir / f"{clipid}.npy"
            if out_path.exists() and not force:
                # skip existing by default
                mapping_rows.append((clipid, str(out_path), getattr(row, "label_block"), getattr(row, "label_prolong"),
                                     getattr(row, "label_soundrep"), getattr(row, "label_wordrep"),
                                     getattr(row, "label_interjection"), getattr(row, "label_no_stutter")))
                continue
            np.save(out_path, feat)
            mapping_rows.append((clipid, str(out_path), getattr(row, "label_block"), getattr(row, "label_prolong"),
                                 getattr(row, "label_soundrep"), getattr(row, "label_wordrep"),
                                 getattr(row, "label_interjection"), getattr(row, "label_no_stutter")))

        except Exception as e:
            errors.append((clipid if 'clipid' in locals() else f"row_{i}", rel_path if 'rel_path' in locals() else "", str(e)))

    # Save mapping CSV
    mapping_df = pd.DataFrame(mapping_rows, columns=["ClipId", "feature_path", "label_block", "label_prolong",
                                                     "label_soundrep", "label_wordrep", "label_interjection", "label_no_stutter"])
    mapping_csv = out_dir / f"mapping_{csv_path.stem}.csv"
    mapping_df.to_csv(mapping_csv, index=False)
    print(f"[INFO] Saved mapping CSV -> {mapping_csv}")

    # Summary
    print(f"[SUMMARY] processed: {len(mapping_rows)} / {total}, errors: {len(errors)}")
    if errors:
        err_path = out_dir / f"errors_{csv_path.stem}.txt"
        with open(err_path, "w") as ef:
            for e in errors:
                ef.write("|".join(map(str, e)) + "\n")
        print(f"[WARN] Some errors occurred. See {err_path}")

    return mapping_df, errors

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract log-mel features from metadata CSVs")
    parser.add_argument("--splits", nargs="+", default=["train","val","test"],
                        help="Which splits to process (default: train val test)")
    parser.add_argument("--metadata_dir", type=str, default="metadata", help="metadata directory")
    parser.add_argument("--features_dir", type=str, default="features", help="output features directory")
    parser.add_argument("--sr", type=int, default=DEFAULT_SR, help="target sample rate")
    parser.add_argument("--n_mels", type=int, default=N_MELS, help="number of mel bands")
    parser.add_argument("--win_ms", type=int, default=WIN_MS, help="window length in ms")
    parser.add_argument("--hop_ms", type=int, default=HOP_MS, help="hop length in ms")
    parser.add_argument("--duration", type=float, default=3.0, help="expected clip duration in seconds")
    parser.add_argument("--force", action="store_true", help="overwrite existing features")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    features_root = Path(args.features_dir)
    ensure_dirs = [features_root]
    for sp in args.splits:
        ensure_dir(features_root / sp)

    for sp in args.splits:
        csv_path = metadata_dir / f"{sp}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found — skipping {sp}")
            continue
        out_dir = features_root / sp
        process_split(csv_path, out_dir, sr=args.sr, n_mels=args.n_mels,
                      win_ms=args.win_ms, hop_ms=args.hop_ms, duration_sec=args.duration, force=args.force)

if __name__ == "__main__":
    main()