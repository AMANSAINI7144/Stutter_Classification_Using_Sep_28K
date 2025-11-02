#!/usr/bin/env python3
# run_inference.py
# Ready-to-paste for project layout where model code lives in src/models/...

import os
import sys
# Ensure src/ is on sys.path so `from models.crnn_attention import CRNN_Attention` works
proj_root = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.join(proj_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import numpy as np
import torch
import soundfile as sf
import librosa

# import model after ensuring src/ is on sys.path
from models.crnn_attention import CRNN_Attention

LABEL_NAMES = [
    "label_block",
    "label_prolong",
    "label_soundrep",
    "label_wordrep",
    "label_interjection",
    "label_no_stutter",
]

def load_audio_and_mels(wav_path,
                        sr=16000,
                        n_mels=80,
                        n_fft=1024,
                        hop_length=256,
                        power=2.0):
    y, sr_loaded = librosa.load(wav_path, sr=sr)
    if y is None or y.size == 0:
        raise RuntimeError(f"Loaded audio is empty: {wav_path}")
    mel = librosa.feature.melspectrogram(y=y,
                                         sr=sr,
                                         n_fft=n_fft,
                                         hop_length=hop_length,
                                         n_mels=n_mels,
                                         power=power)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # (n_mels, T)

def prepare_tensor(log_mel):
    x = np.array(log_mel, dtype=np.float32)
    # add dims -> (1, 1, n_mels, T)
    x = np.expand_dims(x, 0)   # (1, n_mels, T)
    x = np.expand_dims(x, 0)   # (1, 1, n_mels, T)
    # per-sample normalization (same as training)
    mean = x.mean()
    std = x.std() + 1e-8
    x = (x - mean) / std
    tensor = torch.from_numpy(x)
    return tensor

def load_checkpoint_weights(model, ckpt_path, map_location):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    # common keys: 'state_dict', 'model_state', or the raw state_dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            # maybe raw keys already
            state_dict = ckpt
    else:
        state_dict = ckpt

    # strip possible "module." prefixes from DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=True)
    return model

def infer(wav_path, ckpt_path, device):
    log_mel = load_audio_and_mels(wav_path)
    x = prepare_tensor(log_mel)  # (1,1,n_mels,T)
    x = x.to(device)

    model = CRNN_Attention(num_classes=len(LABEL_NAMES), n_mels=log_mel.shape[0])
    # load checkpoint mapping to device (cpu or cuda)
    model = load_checkpoint_weights(model, ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        out, attn = model(x)  # assumes model returns (probabilities, attention) as in your file
        probs = out.squeeze(0).cpu().numpy()
        attn_np = None
        if attn is not None:
            try:
                attn_np = attn.squeeze(0).cpu().numpy()
            except Exception:
                attn_np = attn.cpu().numpy()

    print("\nPredicted probabilities:")
    for name, p in zip(LABEL_NAMES, probs):
        print(f"{name:20s} : {p:.4f}")

    if attn_np is not None:
        # handle shape like (T,1) or (T,)
        if attn_np.ndim == 2 and attn_np.shape[1] == 1:
            attn_np = attn_np.squeeze(1)
        attn_save = wav_path + ".attn.npy"
        np.save(attn_save, attn_np)
        print(f"\nSaved attention weights to: {attn_save} (shape: {attn_np.shape})")

    return probs, attn_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to input wav file")
    parser.add_argument("--checkpoint",
                        default="src/models/checkpoints_attention_models/best_model_attention.pth",
                        help="Path to model checkpoint (default: src/models/checkpoints_attention_models/best_model_attention.pth)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device string for torch (e.g., cpu or cuda)")
    args = parser.parse_args()

    print("WAV:", args.wav)
    print("Checkpoint:", args.checkpoint)
    print("Device:", args.device)

    # quick sanity checks
    if not os.path.exists(args.wav):
        print(f"ERROR: wav file does not exist: {args.wav}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint file does not exist: {args.checkpoint}")
        sys.exit(1)

    infer(args.wav, args.checkpoint, args.device)
