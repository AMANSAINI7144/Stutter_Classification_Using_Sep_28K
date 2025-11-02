# #!/usr/bin/env python3
# """
# Run inference on one WAV, visualize waveform / spectrogram / attention,
# and save everything under audio_visualiser/<wavname>_<timestamp>/
# """

# import os
# import sys
# import shutil
# import json
# from datetime import datetime
# import argparse
# import numpy as np
# import torch
# import librosa
# import soundfile as sf
# import matplotlib.pyplot as plt

# # ensure src/ is on sys.path (so `from models.crnn_attention import CRNN_Attention` works)
# proj_root = os.path.abspath(os.path.dirname(__file__))
# src_path = os.path.join(proj_root, "src")
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# from models.crnn_attention import CRNN_Attention

# LABEL_NAMES = [
#     "label_block",
#     "label_prolong",
#     "label_soundrep",
#     "label_wordrep",
#     "label_interjection",
#     "label_no_stutter",
# ]

# def load_audio(wav_path, sr=16000):
#     y, sr_loaded = librosa.load(wav_path, sr=sr)
#     if y is None or y.size == 0:
#         raise RuntimeError(f"Loaded audio is empty: {wav_path}")
#     return y, sr

# def compute_log_mel(y, sr=16000, n_mels=80, n_fft=1024, hop_length=256, power=2.0):
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
#                                          hop_length=hop_length, n_mels=n_mels, power=power)
#     log_mel = librosa.power_to_db(mel, ref=np.max)
#     return log_mel  # shape (n_mels, T)

# def prepare_tensor(log_mel):
#     x = np.array(log_mel, dtype=np.float32)
#     x = np.expand_dims(x, 0)  # (1, n_mels, T)
#     x = np.expand_dims(x, 0)  # (1, 1, n_mels, T)
#     mean = x.mean(); std = x.std() + 1e-8
#     x = (x - mean) / std
#     tensor = torch.from_numpy(x)
#     return tensor

# def load_checkpoint_weights(model, ckpt_path, map_location):
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
#     ckpt = torch.load(ckpt_path, map_location=map_location)
#     if isinstance(ckpt, dict):
#         if "state_dict" in ckpt:
#             state_dict = ckpt["state_dict"]
#         elif "model_state" in ckpt:
#             state_dict = ckpt["model_state"]
#         else:
#             state_dict = ckpt
#     else:
#         state_dict = ckpt
#     new_state = {}
#     for k, v in state_dict.items():
#         new_k = k[len("module."):] if k.startswith("module.") else k
#         new_state[new_k] = v
#     model.load_state_dict(new_state, strict=True)
#     return model

# def run_model_on_wav(wav_path, ckpt_path, device):
#     # load audio and features
#     y, sr = load_audio(wav_path)
#     log_mel = compute_log_mel(y, sr=sr)
#     x = prepare_tensor(log_mel).to(device)

#     # build model and load ckpt
#     model = CRNN_Attention(num_classes=len(LABEL_NAMES), n_mels=log_mel.shape[0])
#     model = load_checkpoint_weights(model, ckpt_path, map_location=device)
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         out, attn = model(x)
#         probs = out.squeeze(0).cpu().numpy()
#         attn_np = attn.squeeze(0).cpu().numpy() if attn is not None else None

#     return {
#         "y": y,
#         "sr": sr,
#         "log_mel": log_mel,
#         "probs": probs,
#         "attn": attn_np
#     }

# def create_output_dir(wav_path):
#     base = os.path.splitext(os.path.basename(wav_path))[0]
#     t = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_dir = os.path.join(proj_root, "audio_visualiser", f"{base}_{t}")
#     os.makedirs(out_dir, exist_ok=True)
#     return out_dir

# def save_wavecopy(wav_path, out_dir):
#     dest = os.path.join(out_dir, os.path.basename(wav_path))
#     shutil.copy2(wav_path, dest)
#     return dest

# def plot_waveform(y, sr, out_path):
#     plt.figure(figsize=(10, 3))
#     times = np.arange(len(y)) / float(sr)
#     plt.plot(times, y)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title("Waveform")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def plot_spectrogram(log_mel, sr, hop_length, out_path, y_axis="mel"):
#     plt.figure(figsize=(10, 4))
#     # librosa expects shape (n_mels, T)
#     librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length,
#                              x_axis="time", y_axis=y_axis)
#     plt.colorbar(format="%+2.0f dB")
#     plt.title("Log-mel spectrogram (dB)")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def plot_spectrogram_with_attention(log_mel, attn, sr, hop_length, out_path):
#     # log_mel shape: (n_mels, T_spec)
#     T_spec = log_mel.shape[1]
#     # attn may be shorter; interpolate to spectrogram frames
#     if attn is None:
#         raise ValueError("No attention to plot")
#     attn = np.asarray(attn).squeeze()
#     # if attn is 2D with shape (T,1)
#     if attn.ndim == 2 and attn.shape[1] == 1:
#         attn = attn.squeeze(1)
#     # interp to T_spec
#     attn_x = np.linspace(0, T_spec - 1, num=attn.shape[0])
#     attn_interp = np.interp(np.arange(T_spec), attn_x, attn)
#     times = librosa.frames_to_time(np.arange(T_spec), sr=sr, hop_length=hop_length)

#     plt.figure(figsize=(10, 5))
#     ax1 = plt.subplot(2, 1, 1)
#     librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
#     plt.colorbar(format="%+2.0f dB")
#     plt.title("Log-mel spectrogram (dB)")

#     ax2 = plt.subplot(2, 1, 2, sharex=ax1)
#     plt.plot(times, attn_interp)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Attention")
#     plt.title("Model attention (interpolated to spectrogram frames)")
#     plt.tight_layout()
#     plt.savefig(out_path)
#     plt.close()

# def save_results(out_dir, wav_copy_path, sr, probs, attn, log_mel, hop_length):
#     # save numpy arrays
#     np.save(os.path.join(out_dir, "log_mel.npy"), log_mel)
#     if attn is not None:
#         np.save(os.path.join(out_dir, "attention.npy"), attn)

#     # save probabilities as JSON
#     probs_dict = {name: float(p) for name, p in zip(LABEL_NAMES, probs)}
#     meta = {
#         "wav_copy": os.path.basename(wav_copy_path),
#         "sr": int(sr),
#         "probs": probs_dict,
#         "saved_at": datetime.now().isoformat()
#     }
#     with open(os.path.join(out_dir, "results.json"), "w") as f:
#         json.dump(meta, f, indent=2)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--wav", required=True, help="Path to input wav file")
#     parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--hop_length", type=int, default=256, help="hop_length used for mel (frames->time)")
#     args = parser.parse_args()

#     wav_path = args.wav
#     ckpt = args.checkpoint
#     device = args.device

#     print("WAV:", wav_path)
#     print("Checkpoint:", ckpt)
#     print("Device:", device)

#     if not os.path.exists(wav_path):
#         print(f"ERROR: wav file does not exist: {wav_path}")
#         sys.exit(1)
#     if not os.path.exists(ckpt):
#         print(f"ERROR: checkpoint file does not exist: {ckpt}")
#         sys.exit(1)

#     out_dir = create_output_dir(wav_path)
#     wav_copy = save_wavecopy(wav_path, out_dir)
#     print("Created output dir:", out_dir)

#     results = run_model_on_wav(wav_path, ckpt, device)
#     y = results["y"]; sr = results["sr"]; log_mel = results["log_mel"]
#     probs = results["probs"]; attn = results["attn"]

#     # Save raw results
#     save_results(out_dir, wav_copy, sr, probs, attn, log_mel, args.hop_length)

#     # Plots
#     plot_waveform(y, sr, os.path.join(out_dir, "waveform.png"))
#     plot_spectrogram(log_mel, sr, args.hop_length, os.path.join(out_dir, "spectrogram.png"))
#     if attn is not None:
#         try:
#             plot_spectrogram_with_attention(log_mel, attn, sr, args.hop_length,
#                                             os.path.join(out_dir, "spectrogram_with_attention.png"))
#         except Exception as e:
#             print("Warning: failed to plot attention overlay:", e)

#     # print probs summary
#     print("\nPredicted probabilities:")
#     for name, p in zip(LABEL_NAMES, probs):
#         print(f"{name:20s} : {p:.4f}")

#     print("\nSaved visualization and outputs to:", out_dir)
#     print("Files:", os.listdir(out_dir))

# if __name__ == "__main__":
#     main()




import os
import sys
import shutil
import json
from datetime import datetime
import argparse
import numpy as np
import torch
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import csv

# ensure src/ is on sys.path so imports work
proj_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.crnn_attention import CRNN_Attention

LABEL_NAMES = [
    "label_block",
    "label_prolong",
    "label_soundrep",
    "label_wordrep",
    "label_interjection",
    "label_no_stutter",
]

# ---------------------------
# Basic audio / model helpers
# ---------------------------
def load_audio(wav_path, sr=16000):
    y, sr_loaded = librosa.load(wav_path, sr=sr, mono=True)
    if y is None or y.size == 0:
        raise RuntimeError(f"Loaded audio is empty: {wav_path}")
    return y, sr

def compute_log_mel(y, sr=16000, n_mels=80, n_fft=1024, hop_length=256, power=2.0):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # shape (n_mels, T)

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
    # be tolerant to small mismatches
    model.load_state_dict(new_state, strict=False)
    return model

# ---------------------------
# Single-clip inference (exact copy behavior)
# ---------------------------
def run_model_on_wav(wav_path, ckpt_path, device, n_mels=80):
    y, sr = load_audio(wav_path)
    log_mel = compute_log_mel(y, sr=sr, n_mels=n_mels)
    x = prepare_tensor_from_log_mel(log_mel).to(device)

    model = CRNN_Attention(num_classes=len(LABEL_NAMES), n_mels=log_mel.shape[0])
    model = load_checkpoint_weights(model, ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        out, attn = model(x)
        probs = out.squeeze(0).cpu().numpy()
        attn_np = attn.squeeze(0).cpu().numpy() if attn is not None else None

    return {"y": y, "sr": sr, "log_mel": log_mel, "probs": probs, "attn": attn_np}

# ---------------------------
# Sliding-window segmentation
# ---------------------------
def slide_and_infer_and_save(y, sr, model, device,
                             window_sec=2.0, hop_sec=0.5,
                             n_mels=80, n_fft=1024, hop_length=256):
    win_samps = int(round(window_sec * sr))
    hop_samps = int(round(hop_sec * sr))
    audio_len = len(y)
    starts = list(range(0, max(1, audio_len - win_samps + 1), hop_samps))
    # ensure last window covers tail
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
            out, attn = model(x)
            probs = out.squeeze(0).cpu().numpy()
            segments.append({
                "start_s": float(s / sr),
                "end_s": float(min(e, audio_len) / sr),
                "probs": probs.tolist()
            })
    return segments

def save_segments_csv(segments, path):
    header = ["start_s", "end_s"] + LABEL_NAMES
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for seg in segments:
            row = [seg["start_s"], seg["end_s"]] + [seg["probs"][i] for i in range(len(LABEL_NAMES))]
            writer.writerow(row)

def merge_adjacent_segments(segments, threshold=0.3):
    """
    For each window take top label (if above threshold), then merge contiguous windows
    with same top label into intervals.
    Returns list of {label, start_s, end_s, score}
    """
    top_labels = []
    for seg in segments:
        probs = np.array(seg["probs"])
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        if top_prob >= threshold:
            top_labels.append((top_idx, top_prob))
        else:
            top_labels.append((None, float(top_prob)))

    merged = []
    cur_label, cur_score, cur_start, cur_end = None, 0.0, None, None
    for seg, (lab, score) in zip(segments, top_labels):
        s, e = seg["start_s"], seg["end_s"]
        if lab is None:
            # close current if open
            if cur_label is not None:
                merged.append({"label": LABEL_NAMES[cur_label], "start_s": cur_start, "end_s": cur_end, "score": cur_score})
                cur_label, cur_score, cur_start, cur_end = None, 0.0, None, None
            continue
        if cur_label is None:
            cur_label, cur_score, cur_start, cur_end = lab, score, s, e
        elif lab == cur_label:
            # extend
            cur_end = e
            cur_score = max(cur_score, score)
        else:
            # push and start new
            merged.append({"label": LABEL_NAMES[cur_label], "start_s": cur_start, "end_s": cur_end, "score": cur_score})
            cur_label, cur_score, cur_start, cur_end = lab, score, s, e
    if cur_label is not None:
        merged.append({"label": LABEL_NAMES[cur_label], "start_s": cur_start, "end_s": cur_end, "score": cur_score})
    return merged

# ---------------------------
# Plot helpers
# ---------------------------
def plot_waveform(y, sr, out_path):
    plt.figure(figsize=(10, 3))
    times = np.arange(len(y)) / float(sr)
    plt.plot(times, y, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram(log_mel, sr, hop_length, out_path):
    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax)
    plt.colorbar(img, format="%+2.0f dB", ax=ax)
    plt.title("Log-mel spectrogram (dB)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram_with_attention(log_mel, attn, sr, hop_length, out_path):
    if attn is None:
        raise ValueError("No attention to plot")
    T_spec = log_mel.shape[1]
    attn = np.asarray(attn).squeeze()
    if attn.ndim == 2 and attn.shape[1] == 1:
        attn = attn.squeeze(1)
    attn_x = np.linspace(0, T_spec - 1, num=attn.shape[0])
    attn_interp = np.interp(np.arange(T_spec), attn_x, attn)
    times = librosa.frames_to_time(np.arange(T_spec), sr=sr, hop_length=hop_length)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax1)
    plt.colorbar(img, format="%+2.0f dB", ax=ax1)
    ax1.set_title("Log-mel spectrogram (dB)")
    ax2.plot(times, attn_interp)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Attention")
    ax2.set_title("Model attention (interpolated to spectrogram frames)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_labels_timeline(log_mel_full, sr, hop_length, segments, out_path, threshold=0.3):
    # Determine top label per window (or None)
    times = []
    label_indices = []
    for seg in segments:
        s, e = seg["start_s"], seg["end_s"]
        probs = np.array(seg["probs"])
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        if top_prob >= threshold:
            label_indices.append(top_idx)
        else:
            label_indices.append(-1)
        times.append((s, e))

    fig, (ax_spec, ax_lbl) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios":[4,1]})
    img = librosa.display.specshow(log_mel_full, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax_spec)
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

# ---------------------------
# I/O helpers
# ---------------------------
def create_output_dir(wav_path):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(proj_root, "audio_visualiser", f"{base}_{t}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_wavecopy(wav_path, out_dir):
    dest = os.path.join(out_dir, os.path.basename(wav_path))
    shutil.copy2(wav_path, dest)
    return dest

def save_results_json(out_dir, wav_copy_path, sr, probs, attn):
    probs_dict = {name: float(p) for name, p in zip(LABEL_NAMES, probs)}
    meta = {"wav_copy": os.path.basename(wav_copy_path), "sr": int(sr), "probs": probs_dict, "saved_at": datetime.now().isoformat()}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(meta, f, indent=2)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to input wav file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hop_length", type=int, default=256, help="hop_length used for mel (frames->time)")
    parser.add_argument("--window_sec", type=float, default=2.0, help="sliding window size (s)")
    parser.add_argument("--hop_sec", type=float, default=0.5, help="sliding hop size (s)")
    parser.add_argument("--threshold", type=float, default=0.3, help="prob threshold to mark label present")
    args = parser.parse_args()

    wav_path = args.wav
    ckpt = args.checkpoint
    device = args.device
    print("WAV:", wav_path)
    print("Checkpoint:", ckpt)
    print("Device:", device)

    if not os.path.exists(wav_path):
        print(f"ERROR: wav file does not exist: {wav_path}"); sys.exit(1)
    if not os.path.exists(ckpt):
        print(f"ERROR: checkpoint file does not exist: {ckpt}"); sys.exit(1)

    out_dir = create_output_dir(wav_path)
    wav_copy = save_wavecopy(wav_path, out_dir)
    print("Created output dir:", out_dir)

    # run single-clip inference (original behavior)
    results = run_model_on_wav(wav_path, ckpt, device)
    y = results["y"]; sr = results["sr"]; log_mel = results["log_mel"]
    probs = results["probs"]; attn = results["attn"]

    # save basic outputs
    np.save(os.path.join(out_dir, "log_mel.npy"), log_mel)
    if attn is not None:
        np.save(os.path.join(out_dir, "attention.npy"), attn)
    save_results_json(out_dir, wav_copy, sr, probs, attn)

    # plots: waveform, spectrogram, attention overlay
    plot_waveform(y, sr, os.path.join(out_dir, "waveform.png"))
    plot_spectrogram(log_mel, sr, args.hop_length, os.path.join(out_dir, "spectrogram.png"))
    if attn is not None:
        try:
            plot_spectrogram_with_attention(log_mel, attn, sr, args.hop_length, os.path.join(out_dir, "spectrogram_with_attention.png"))
        except Exception as e:
            print("Warning: failed to plot attention overlay:", e)

    # ---------------------------
    # Sliding-window segmentation & timeline
    # ---------------------------
    model = CRNN_Attention(num_classes=len(LABEL_NAMES), n_mels=log_mel.shape[0])
    model = load_checkpoint_weights(model, ckpt, map_location=device)
    model.to(device)

    segments = slide_and_infer_and_save(y, sr, model, device,
                                        window_sec=args.window_sec, hop_sec=args.hop_sec,
                                        n_mels=log_mel.shape[0], n_fft=1024, hop_length=args.hop_length)

    # save raw segments.csv
    import csv as _csv
    seg_csv = os.path.join(out_dir, "segments.csv")
    header = ["start_s", "end_s"] + LABEL_NAMES
    with open(seg_csv, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        for seg in segments:
            writer.writerow([seg["start_s"], seg["end_s"]] + [seg["probs"][i] for i in range(len(LABEL_NAMES))])
    print("Saved segments CSV:", seg_csv)

    # merge adjacent windows with same label above threshold
    merged = merge_adjacent_segments(segments, threshold=args.threshold)
    merged_csv = os.path.join(out_dir, "merged_segments.csv")
    with open(merged_csv, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["label", "start_s", "end_s", "score"])
        for m in merged:
            writer.writerow([m["label"], m["start_s"], m["end_s"], m["score"]])
    print("Saved merged segments CSV:", merged_csv)

    # timeline + heatmap plots
    # compute full log_mel for the whole file for plotting timeline
    log_mel_full = compute_log_mel(y, sr=sr, n_mels=log_mel.shape[0], n_fft=1024, hop_length=args.hop_length)
    plot_labels_timeline(log_mel_full, sr, args.hop_length, segments, os.path.join(out_dir, "labels_timeline.png"), threshold=args.threshold)
    plot_probs_heatmap(segments, os.path.join(out_dir, "probs_heatmap.png"))

    # final probs summary
    print("\nPredicted probabilities (whole-clip):")
    for name, p in zip(LABEL_NAMES, probs):
        print(f"{name:20s} : {p:.4f}")

    print("\nSaved visualization and segmentation outputs to:", out_dir)
    print("Files:", os.listdir(out_dir))

if __name__ == "__main__":
    main()