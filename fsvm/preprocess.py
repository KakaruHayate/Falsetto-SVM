"""MFCC preprocessing for every wav under <data/audio_dir>.

For each wav we save:
  data/cache_mfcc/<name>.npy  -- shape (mfcc_order, n_frames)  (e.g. 13 x ~25)
  data/cache_label/<name>.npy -- shape (n_frames,) with -1 (chest) or +1 (falsetto)

This script replaces the original Falsetto-SVM/preprocess.py with two fixes:
  - hop_length = 882  (paper: 50 Hz frame rate at sr=44.1 kHz, "undersampled" design)
  - works on a single shared audio folder; the train/val split is decided later
    by song identity (sing_id.csv) -- not random shuffling.
"""
from __future__ import annotations

import argparse
import os
import sys

import librosa
import numpy as np

from utils import DotDict, ensure_dir, load_config, traverse_dir


def parse_args() -> DotDict:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()


def _frame_count(samples: int, fft_size: int, hop_length: int) -> int:
    # librosa: n_frames = 1 + (n - n_fft) // hop_length, when center=True (default).
    return 1 + (max(1, samples) - fft_size) // hop_length


def mfcc_for_wav(
    path: str,
    sampling_rate: int,
    mfcc_order: int,
    fft_size: int,
    hop_length: int,
    pre_emph: float,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    audio, _sr = librosa.load(path, sr=sampling_rate, mono=True)
    audio = librosa.effects.preemphasis(audio, coef=pre_emph)
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sampling_rate,
        n_mfcc=mfcc_order,
        n_fft=fft_size,
        hop_length=hop_length,
        # default center=True → mirrors paper behaviour for 1 + (N-fft)//hop
    )
    mfcc_frames = mfcc.shape[1]
    if target_frames is not None:
        if mfcc_frames < target_frames:
            # repeat last frame to pad
            pad = mfcc[:, -1:].repeat(target_frames - mfcc_frames, axis=1)
            mfcc = np.concatenate([mfcc, pad], axis=1)
        elif mfcc_frames > target_frames:
            mfcc = mfcc[:, :target_frames]
    # labels
    base = os.path.basename(path).rsplit(".", 1)[0]
    label_value = -1 if "chest" in base else 1
    label = np.full(mfcc.shape[1], label_value, dtype=np.float32)
    return mfcc.astype(np.float32), label


def main() -> int:
    cfg_path = parse_args().config
    cfg = load_config(cfg_path)

    audio_dir = cfg.data.audio_dir
    parent = os.path.dirname(audio_dir) or "."
    mfcc_out = os.path.join(parent, "mfcc")
    label_out = os.path.join(parent, "label")
    for d in (mfcc_out, label_out):
        ensure_dir(d)

    files = traverse_dir(audio_dir, extension="wav", is_sort=True, is_ext=True)
    print(f"[preprocess] {len(files)} wavs in {audio_dir}")
    print(f"[preprocess] hop={cfg.data.hop_length}  fft={cfg.data.fft_size}  "
          f"target_frames={cfg.data.train_frames}")
    for rel in files:
        # rel includes ".wav" because is_ext=True
        base = rel[:-4] if rel.endswith(".wav") else rel
        src = os.path.join(audio_dir, base + ".wav")
        name = base.replace("/", os.sep)
        dst_m = os.path.join(mfcc_out, name + ".npy")
        dst_l = os.path.join(label_out, name + ".npy")
        if os.path.exists(dst_m) and os.path.exists(dst_l):
            continue
        mfcc, lab = mfcc_for_wav(
            src,
            sampling_rate=cfg.data.sampling_rate,
            mfcc_order=cfg.data.mfcc_order,
            fft_size=cfg.data.fft_size,
            hop_length=cfg.data.hop_length,
            pre_emph=cfg.data.pre_emph,
            target_frames=cfg.data.train_frames,
        )
        np.save(dst_m, mfcc)
        np.save(dst_l, lab)
    print(f"[preprocess] done; mfccs -> {mfcc_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
