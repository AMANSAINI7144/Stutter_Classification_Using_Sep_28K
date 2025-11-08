#!/usr/bin/env python3
"""
Export ModelWithPreproc -> TorchScript

Usage examples:

# Export (1 second example input)
cd /home/aman2/projects/MPC_Project/Mobile
python3 export_to_torchscript.py \
  --ckpt ../models/checkpoints/attention_expt1/best_model_attention.pth \
  --out out/model_torchscript.pt \
  --example-length-sec 1

# Export and run quick parity test with a real WAV
python3 export_to_torchscript.py \
  --ckpt ../models/checkpoints/attention_expt1/best_model_attention.pth \
  --out out/model_torchscript.pt \
  --example-length-sec 1 \
  --test-wav ../data/SEP28K/clips/stuttering-clips/clips/FluencyBank_010_11.wav
"""

import os
import sys
import argparse
import torch
import numpy as np

# ensure project src is importable (export_wrapper already does this too, but safe here)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# import wrapper building utilities from export_wrapper.py
from export_wrapper import ModelWithPreproc, load_checkpoint_into_model, DEFAULT_SR

# For WAV loading in test mode
try:
    import soundfile as sf
    import librosa
except Exception:
    sf = None
    librosa = None


def export_torchscript(ckpt_path: str, out_path: str, example_length_sec: float = 1.0, device: str = "cpu"):
    """
    Build wrapper (loads checkpoint inside), trace the wrapper with a dummy waveform tensor,
    and save the TorchScript module to out_path.
    """
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print("Device for export:", device_t)

    # load base model weights into CRNN_Attention (using helper)
    print("Loading checkpoint into base model:", ckpt_path)
    base_model = load_checkpoint_into_model(ckpt_path, device=device_t)

    # create wrapper
    wrapper = ModelWithPreproc(base_model,
                               sr=DEFAULT_SR,
                               n_mels=wrapper_n_mels_from_base(base_model),
                               n_fft=1024,
                               hop_length=256,
                               power=2.0)
    wrapper.to(device_t)
    wrapper.eval()

    # Create dummy waveform tensor for tracing: shape [1, L]
    L = int(round(example_length_sec * DEFAULT_SR))
    dummy = torch.randn(1, L, dtype=torch.float32).to(device_t)

    print(f"Tracing wrapper with dummy input shape: {tuple(dummy.shape)} (seconds={example_length_sec})")

    # Try to trace; if trace fails, try script as fallback
    ts_module = None
    try:
        ts_module = torch.jit.trace(wrapper, dummy, strict=False)
        print("Tracing succeeded.")
    except Exception as e:
        print("Tracing failed with error:", e)
        print("Attempting torch.jit.script(...) fallback...")
        try:
            ts_module = torch.jit.script(wrapper)
            print("Scripting succeeded.")
        except Exception as e2:
            print("Scripting also failed. Aborting. Error:", e2)
            raise RuntimeError("Could not produce TorchScript module via trace or script.") from e2

    # ensure output dir exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # save
    ts_module.save(out_path)
    print("Saved TorchScript model to:", out_path)
    return out_path


def wrapper_n_mels_from_base(base_model):
    """
    Try to guess n_mels for wrapper from base model attributes (many CRNNs store n_mels or input shape).
    Fallback to DEFAULT_N_MELS (80) if not found.
    """
    # try common attributes
    for attr in ["n_mels", "num_mels", "n_mel"]:
        if hasattr(base_model, attr):
            return getattr(base_model, attr)
    # sometimes base model has conv1 weight with shape -> infer channels
    try:
        w = None
        for k, v in base_model.state_dict().items():
            if "conv" in k and v.ndim >= 2:
                w = v
                break
        if w is not None:
            # not reliable, fallback to 80
            return 80
    except Exception:
        pass
    return 80


def load_wav_to_tensor(wav_path: str, target_sr: int = DEFAULT_SR):
    if sf is None:
        raise RuntimeError("soundfile is required for test mode. Install with `pip install soundfile librosa`.")
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype("float32")
    if sr != target_sr:
        if librosa is None:
            raise RuntimeError("librosa required for resampling in test mode.")
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    # return tensor shape [1, L]
    return torch.from_numpy(data).unsqueeze(0)


def run_parity_test(ckpt_path: str, ts_path: str, wav_path: str, device: str = "cpu"):
    """
    Run wrapper model (Python) and TorchScript model on same wav and print outputs.
    """
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print("Running parity test on device:", device_t)
    print("Loading base model from checkpoint:", ckpt_path)
    base_model = load_checkpoint_into_model(ckpt_path, device=device_t)
    wrapper = ModelWithPreproc(base_model,
                               sr=DEFAULT_SR,
                               n_mels=wrapper_n_mels_from_base(base_model),
                               n_fft=1024,
                               hop_length=256,
                               power=2.0)
    wrapper.to(device_t)
    wrapper.eval()

    print("Loading TorchScript model:", ts_path)
    ts = torch.jit.load(ts_path, map_location=device_t)
    ts.eval()

    print("Loading WAV:", wav_path)
    wav_t = load_wav_to_tensor(wav_path, target_sr=DEFAULT_SR).to(device_t)

    with torch.no_grad():
        out_py, attn_py = wrapper(wav_t)
        out_ts, attn_ts = ts(wav_t)

    out_py_np = out_py.squeeze(0).cpu().numpy()
    out_ts_np = out_ts.squeeze(0).cpu().numpy()

    print("\nPython wrapper outputs:")
    for i, p in enumerate(out_py_np):
        print(f"  idx {i:02d} : {p:.6f}")

    print("\nTorchScript outputs:")
    for i, p in enumerate(out_ts_np):
        print(f"  idx {i:02d} : {p:.6f}")

    diff = np.abs(out_py_np - out_ts_np)
    print("\nAbsolute differences per class:")
    for i, d in enumerate(diff):
        print(f"  idx {i:02d} : {d:.6e}")
    print("Max abs diff:", diff.max())


def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Export ModelWithPreproc to TorchScript")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--out", required=True, help="Path to save TorchScript model (.pt)")
    parser.add_argument("--example-length-sec", type=float, default=1.0, help="Example waveform length in seconds used for tracing")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for export (cpu recommended)")
    parser.add_argument("--test-wav", required=False, help="Optional path to wav for parity test after export")
    args = parser.parse_args()

    out_path = export_torchscript(args.ckpt, args.out, example_length_sec=args.example_length_sec, device=args.device)
    if args.test_wav:
        run_parity_test(args.ckpt, out_path, args.test_wav, device=args.device)


if __name__ == "__main__":
    parse_args_and_run()
