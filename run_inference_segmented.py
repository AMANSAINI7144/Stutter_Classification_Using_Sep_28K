import os
import sys
import argparse
import json
import shutil
from datetime import datetime

import numpy as np
import librosa
import librosa.display
import torch
import matplotlib.pyplot as plt
import csv

# ensure src/ is on sys.path
proj_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.crnn_attention import CRNN_Attention  # adjust if module name differs

LABEL_NAMES = [
    "label_block",
    "label_prolong",
    "label_soundrep",
    "label_wordrep",
    "label_interjection",
    "label_no_stutter",
]

def load_audio(wav_path, sr=16000):
    y, sr_loaded = librosa.load(wav_path, sr=sr)
    if y is None or y.size == 0:
        raise RuntimeError(f"Loaded audio is empty: {wav_path}")
    return y, sr

def compute_log_mel(y, sr, n_mels=80, n_fft=1024, hop_length=256, power=2.0):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def prepare_tensor_from_log_mel(log_mel):
    x = np.array(log_mel, dtype=np.float32)
    x = np.expand_dims(x, 0)  # (1, n_mels, T)
    x = np.expand_dims(x, 0)  # (1, 1, n_mels, T)
    mean = x.mean(); std = x.std() + 1e-8
    x = (x - mean) / std
    return torch.from_numpy(x)

def load_checkpoint_weights(model, ckpt_path, map_location):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    new_state = {}
    for k, v in state_dict.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_state[new_k] = v
    # Use strict=False so minor mismatches don't crash (e.g., extra keys)
    model.load_state_dict(new_state, strict=False)
    return model

def slide_and_infer(y, sr, model, device,
                    window_sec=2.0, hop_sec=0.5,
                    n_mels=80, n_fft=1024, hop_length=256):
    win_samps = int(round(window_sec * sr))
    hop_samps = int(round(hop_sec * sr))
    audio_len = len(y)
    starts = list(range(0, max(1, audio_len - win_samps + 1), hop_samps))
    if (audio_len - win_samps) > 0 and (audio_len - (starts[-1] + win_samps)) > 0:
        starts.append(audio_len - win_samps)
    segments = []
    model.eval()
    with torch.no_grad():
        for s in starts:
            e = s + win_samps
            seg = y[s:e] if e <= audio_len else np.pad(y[s:audio_len], (0, e-audio_len))
            log_mel = compute_log_mel(seg, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
            x = prepare_tensor_from_log_mel(log_mel).to(device)
            out, attn = model(x)  # out: [1, num_classes]
            probs = out.squeeze(0).cpu().numpy()
            segments.append({
                "start_s": float(s / sr),
                "end_s": float(min(e, audio_len) / sr),
                "probs": probs.tolist()
            })
    return segments

def create_output_dir(wav_path, tag="segmented"):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(proj_root, "audio_visualiser", f"{base}_{tag}_{t}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_segments_csv(segments, out_csv_path):
    header = ["start_s", "end_s"] + LABEL_NAMES
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for seg in segments:
            row = [seg["start_s"], seg["end_s"]] + [seg["probs"][i] for i in range(len(LABEL_NAMES))]
            writer.writerow(row)

def plot_waveform(y, sr, out_path):
    times = np.arange(len(y)) / float(sr)
    plt.figure(figsize=(12, 3))
    plt.plot(times, y, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram(log_mel_full, sr, hop_length, out_path):
    plt.figure(figsize=(12, 4))
    ax = plt.gca()
    img = librosa.display.specshow(
        log_mel_full, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax
    )
    plt.colorbar(img, format="%+2.0f dB", ax=ax)
    plt.title("Log-mel spectrogram (dB)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_labels_timeline(log_mel_full, sr, hop_length, segments, out_path, threshold=0.3):
    # Prepare predicted top label per segment (or "none")
    times = []
    label_indices = []
    for seg in segments:
        s = seg["start_s"]
        e = seg["end_s"]
        probs = np.array(seg["probs"])
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        if top_prob >= threshold:
            label_indices.append(top_idx)
        else:
            label_indices.append(-1)
        times.append((s, e))

    # Plot with two subplots: spectrogram (upper) and labels (lower)
    fig, (ax_spec, ax_lbl) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [4, 1]})
    img = librosa.display.specshow(log_mel_full, sr=sr, hop_length=hop_length,
                                   x_axis="time", y_axis="mel", ax=ax_spec)
    plt.colorbar(img, format="%+2.0f dB", ax=ax_spec)
    ax_spec.set_title("Log-mel spectrogram (dB)")

    cmap = plt.get_cmap("tab10")
    ax_lbl.set_ylim(0, 1)
    ax_lbl.set_yticks([])
    total_dur = librosa.frames_to_time(log_mel_full.shape[1], sr=sr, hop_length=hop_length)
    for (s, e), lab in zip(times, label_indices):
        if lab == -1:
            color = "lightgray"
            label_text = ""
        else:
            color = cmap(lab % 10)
            label_text = LABEL_NAMES[lab]
        ax_lbl.barh(0.5, width=(e - s), left=s, height=1.0, color=color, edgecolor="k", alpha=0.85)
        if label_text:
            ax_lbl.text((s + e) / 2.0, 0.5, label_text, ha="center", va="center", fontsize=8, color="white")
    ax_lbl.set_xlim(0, total_dur)
    ax_lbl.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_probs_heatmap(segments, out_path):
    probs_mat = np.stack([np.array(seg["probs"]) for seg in segments], axis=1)  # (labels, windows)
    plt.figure(figsize=(12, 3))
    im = plt.imshow(probs_mat, aspect="auto", origin="lower", interpolation="nearest")
    plt.yticks(ticks=np.arange(len(LABEL_NAMES)), labels=LABEL_NAMES)
    plt.colorbar(im, label="probability")
    plt.xlabel("Window index")
    plt.title("Label probabilities over windows")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window_sec", type=float, default=2.0, help="window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=0.5, help="hop size in seconds")
    parser.add_argument("--threshold", type=float, default=0.3, help="prob threshold to mark label as present")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    wav_path = args.wav
    ckpt = args.checkpoint
    device = args.device

    print("WAV:", wav_path)
    print("CHECKPOINT:", ckpt)
    print("DEVICE:", device)
    print("WINDOW_SEC:", args.window_sec, "HOP_SEC:", args.hop_sec, "THRESH:", args.threshold)

    if not os.path.exists(wav_path):
        print("ERROR: wav file not found:", wav_path); sys.exit(1)
    if not os.path.exists(ckpt):
        print("ERROR: checkpoint not found:", ckpt); sys.exit(1)

    out_dir = create_output_dir(wav_path, tag="segmented")
    shutil.copy2(wav_path, os.path.join(out_dir, os.path.basename(wav_path)))

    y, sr = load_audio(wav_path, sr=16000)
    model = CRNN_Attention(num_classes=len(LABEL_NAMES), n_mels=args.n_mels)
    model = load_checkpoint_weights(model, ckpt, map_location=device)
    model.to(device)

    segments = slide_and_infer(y, sr, model, device,
                               window_sec=args.window_sec, hop_sec=args.hop_sec,
                               n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)

    csv_path = os.path.join(out_dir, "segments.csv")
    save_segments_csv(segments, csv_path)

    summary = {
        "wav": os.path.basename(wav_path),
        "n_segments": len(segments),
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "threshold": args.threshold,
        "labels": LABEL_NAMES
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_waveform(y, sr, os.path.join(out_dir, "waveform.png"))
    log_mel_full = compute_log_mel(y, sr, n_mels=args.n_mels, n_fft=args.n_fft, hop_length=args.hop_length)
    plot_spectrogram(log_mel_full, sr, args.hop_length, os.path.join(out_dir, "spectrogram.png"))
    plot_labels_timeline(log_mel_full, sr, args.hop_length, segments, os.path.join(out_dir, "labels_timeline.png"), threshold=args.threshold)
    plot_probs_heatmap(segments, os.path.join(out_dir, "probs_heatmap.png"))

    print("Saved outputs to:", out_dir)
    print("CSV timeline:", csv_path)

if __name__ == "__main__":
    main()