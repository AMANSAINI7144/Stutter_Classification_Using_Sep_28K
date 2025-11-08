#!/usr/bin/env python3
"""
export_wrapper.py

Defines ModelWithPreproc which wraps your CRNN_Attention model plus
preprocessing (mel spectrogram, dB conversion, per-sample normalization).

This wrapper is intended to be used by an exporter script (export_to_torchscript.py)
which will call torch.jit.trace/script and save a single model_torchscript.pt artifact.

This file also provides a small CLI test: given a checkpoint and a wav file,
it will load the wrapper (with weights) and run the model on the wav and print
predicted probabilities.

Usage (test):
  python export_wrapper.py \
    --ckpt /home/aman2/projects/MPC_Project/models/checkpoints/attention_expt1/best_model_attention.pth \
    --wav /home/aman2/projects/MPC_Project/data/SEP28K/clips/...FluencyBank_010_11.wav

Notes:
- This wrapper assumes input audio sample rate = 16000 Hz (16 kHz).
- It uses torchaudio transforms (MelSpectrogram + AmplitudeToDB) which are scriptable.
- The wrapper expects the base model (CRNN_Attention) to accept input of shape (B,1,n_mels,T)
  and to return (out, attn) similar to your training/inference code.
"""

import os
import sys
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn

# Try to import torchaudio; export to TorchScript relies on torchaudio transforms
try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
except Exception as e:
    torchaudio = None
    MelSpectrogram = None
    AmplitudeToDB = None

# For CLI-only wav loading (not used during tracing); librosa/soundfile ok here
try:
    import soundfile as sf
    import numpy as np
except Exception:
    sf = None

# Ensure repo src/ is on path so we can import CRNN_Attention
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Import your model class
try:
    from models.crnn_attention import CRNN_Attention
except Exception as e:
    # help the user understand why import failed
    raise ImportError(
        "Failed to import CRNN_Attention from src/models/crnn_attention.py. "
        "Make sure this file exists and that `SRC_PATH` is correct. "
        f"Original error: {e}"
    )


# Default preprocessing params (match training/inference scripts)
DEFAULT_SR = 16000
DEFAULT_N_MELS = 80
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256
DEFAULT_POWER = 2.0
EPS = 1e-8


class ModelWithPreproc(nn.Module):
    """
    Wrapper module: accepts raw waveform tensor (shape: [L] or [B, L], dtype float)
    and returns (out, attn) where out is model probabilities tensor.

    Preprocessing steps:
      - expect input sampled at 16000 Hz (caller must ensure or resample beforehand)
      - compute MelSpectrogram (n_fft, hop_length, n_mels, power)
      - convert amplitude -> dB (log-mel)
      - per-sample normalization: (x - mean) / (std + eps)
      - add channel dim to get (B,1,n_mels,T) and forward to base model
    """

    def __init__(self,
                 base_model: nn.Module,
                 sr: int = DEFAULT_SR,
                 n_mels: int = DEFAULT_N_MELS,
                 n_fft: int = DEFAULT_N_FFT,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 power: float = DEFAULT_POWER):
        super(ModelWithPreproc, self).__init__()

        if torchaudio is None or MelSpectrogram is None or AmplitudeToDB is None:
            raise RuntimeError("torchaudio is required for ModelWithPreproc. "
                               "Install torchaudio and try again.")

        # store base model (CRNN_Attention instance)
        self.base = base_model

        # preprocessing transforms (scriptable)
        # MelSpectrogram returns power spectrogram if power=2.0
        self.mel_spec = MelSpectrogram(sample_rate=sr,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       power=power)
        self.amp_to_db = AmplitudeToDB(stype="power")

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        waveform: Tensor of shape [L] or [B, L] (float32) with values in typical PCM float range.
        Returns: (out, attn) where out shape is [B, num_classes]
        """
        # ensure float tensor
        if waveform.dtype != torch.float32:
            waveform = waveform.float()

        # make batch dim if necessary
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, L]

        # waveform expected shape -> [B, L]
        # mel_spec expects shape [B, L] or [L]? torchaudio's MelSpectrogram accepts [B, L]
        mel = self.mel_spec(waveform)  # -> [B, n_mels, T]
        log_mel = self.amp_to_db(mel)  # -> [B, n_mels, T]

        # per-sample normalization across n_mels and time
        # compute mean/std per batch item
        # mean shape = [B, 1, 1]
        mean = log_mel.mean(dim=(1, 2), keepdim=True)
        std = log_mel.std(dim=(1, 2), keepdim=True)
        norm = (log_mel - mean) / (std + EPS)  # [B, n_mels, T]

        # add channel dim: expects (B,1,n_mels,T)
        x = norm.unsqueeze(1)

        # forward through base model
        out, attn = self.base(x)
        return out, attn


# ----- Helper: load checkpoint into CRNN_Attention instance -----
def load_checkpoint_into_model(ckpt_path: str,
                               device: torch.device = torch.device("cpu"),
                               num_classes: int = 6,
                               n_mels: int = DEFAULT_N_MELS) -> nn.Module:
    """
    Create CRNN_Attention, load checkpoint weights (with common key handling),
    and return the model (in eval mode).
    """

    # instantiate base model
    base = CRNN_Attention(num_classes=num_classes, n_mels=n_mels)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    # handle various checkpoint shapes: raw state_dict / dict with 'state_dict' or 'model_state'
    if isinstance(state, dict):
        if "state_dict" in state:
            state_dict = state["state_dict"]
        elif "model_state" in state:
            state_dict = state["model_state"]
        else:
            state_dict = state
    else:
        state_dict = state

    # strip module. prefix if present (DataParallel)
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = k[len("module."):]
        new_state[new_k] = v

    # load with strict=False to be tolerant to small mismatches
    base.load_state_dict(new_state, strict=False)
    base.to(device)
    base.eval()
    return base


# ----- Minimal CLI test so you can verify wrapper behavior -----
def load_wavefile(wav_path: str, target_sr: int = DEFAULT_SR):
    """
    Load a wav file to numpy and return a 1-D float32 numpy array sampled at target_sr.
    This function uses soundfile (sf) where available. If the wav is not at target_sr,
    we will resample using torchaudio (if available) or using numpy/simple method.
    """
    if sf is None:
        raise RuntimeError("soundfile (pysoundfile) is required for CLI wav loading. "
                           "Install via `pip install soundfile`.")

    data, sr = sf.read(wav_path)
    # convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    # convert to float32
    data = data.astype("float32")

    # resample if necessary (use torchaudio if installed)
    if sr != target_sr:
        if torchaudio is not None:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = torch.from_numpy(data).unsqueeze(0)  # [1, L]
            waveform = resampler(waveform).squeeze(0).numpy()
            return waveform
        else:
            # naive resample fallback using numpy (not high-quality) - only as last resort
            import numpy as np
            duration = data.shape[0] / sr
            new_len = int(round(duration * target_sr))
            waveform = np.interp(
                np.linspace(0.0, duration, new_len, endpoint=False),
                np.linspace(0.0, duration, data.shape[0], endpoint=False),
                data
            ).astype("float32")
            return waveform
    return data


def cli_main():
    parser = argparse.ArgumentParser(description="Test ModelWithPreproc wrapper (no tracing).")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--wav", required=False, help="Path to wav file for a quick run/test")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run test on")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of output classes")
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, help="n_mels for mel spectrogram")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("Using device:", device)

    # load base model weights
    print("Loading checkpoint:", args.ckpt)
    base_model = load_checkpoint_into_model(args.ckpt, device=device, num_classes=args.num_classes, n_mels=args.n_mels)

    # create wrapper
    wrapper = ModelWithPreproc(base_model,
                               sr=DEFAULT_SR,
                               n_mels=args.n_mels,
                               n_fft=DEFAULT_N_FFT,
                               hop_length=DEFAULT_HOP_LENGTH,
                               power=DEFAULT_POWER)
    wrapper.to(device)
    wrapper.eval()

    if args.wav:
        if not os.path.exists(args.wav):
            print("WAV not found:", args.wav)
            return
        print("Loading WAV:", args.wav)
        wav_np = load_wavefile(args.wav, target_sr=DEFAULT_SR)
        # convert to torch tensor
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).to(device)  # shape [1, L]

        with torch.no_grad():
            out, attn = wrapper(wav_t)   # out: [B, num_classes]
            probs = out.squeeze(0).cpu().numpy()
            print("\nPredicted probabilities (wrapper):")
            for i, p in enumerate(probs):
                print(f"  class_{i:02d} : {p:.6f}")
            if attn is not None:
                print("  (attention shape):", attn.shape)
    else:
        # quick smoke test with random waveform (1 sec)
        print("No WAV provided - running smoke test with random waveform (1 sec).")
        L = DEFAULT_SR
        rand = torch.randn(1, L).to(device)
        with torch.no_grad():
            out, attn = wrapper(rand)
            print("Output shape:", out.shape)
            print("Attention:", None if attn is None else attn.shape)


if __name__ == "__main__":
    cli_main()
