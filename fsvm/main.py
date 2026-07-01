"""Singer-aware inference for a single short wav.

Usage:
    python main.py -i some_clip.wav [-m results/models] [-c config.yaml]

Pipeline:
    1. Compute the 256-d d-vector for the input wav.
    2. Read the singer-id embeddings (1280 x 256) produced by singer_id.py,
       compute per-singer centroid, and choose the nearest singer by cosine.
    3. Run MFCC preprocessing on the input wav (same config as training) and
       feed the resulting (25, 13) feature matrix into the per-singer linear
       and RBF SVMs. Output per-kernel frame verdicts and a clip-level verdict:
        falsetto = >50% of frames classified as +1.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import librosa
import joblib
import numpy as np
import soundfile as sf
import torch

from resemblyzer import VoiceEncoder

from utils import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-m", "--models_dir",
                   default=r"results/falsetto/models")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--centroids", default="data/singer_id/embeddings.npy",
                   help="path to the singer embeddings matrix")
    p.add_argument("--singer_csv", default="data/singer_id/singer_id.csv")
    return p.parse_args()


def _read_wav(path: str):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def _embed(wav_path: str, encoder: VoiceEncoder, emb_sr: int) -> np.ndarray:
    wav, sr = _read_wav(wav_path)
    if sr != emb_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=emb_sr)
    with torch.no_grad():
        return encoder.embed_utterance(wav)


def _mfcc_features(
    wav_path: str, sampling_rate: int, mfcc_order: int, fft_size: int,
    hop_length: int, pre_emph: float, target_frames: int,
) -> np.ndarray:
    audio, _ = librosa.load(wav_path, sr=sampling_rate, mono=True)
    audio = librosa.effects.preemphasis(audio, coef=pre_emph)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sampling_rate, n_mfcc=mfcc_order,
        n_fft=fft_size, hop_length=hop_length,
    )
    T = mfcc.shape[1]
    if T < target_frames:
        pad = mfcc[:, -1:].repeat(target_frames - T, axis=1)
        mfcc = np.concatenate([mfcc, pad], axis=1)
    elif T > target_frames:
        mfcc = mfcc[:, :target_frames]
    # transpose to (n_frames, 13) for sklearn
    return mfcc.T.astype(np.float32)


def _cosine_sim_matrix(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """a (D,), B (N, D). returns cosine sims of a against each row of B."""
    a_norm = a / max(np.linalg.norm(a), 1e-12)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
    return B_norm @ a_norm


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    os.chdir(os.path.dirname(os.path.abspath(__file__))
             if os.path.isfile(args.config) else ".")

    # -- Step 1: embedding -----------------------------------------------
    weights_path = None if cfg.singer_id.encoder_weights == "bundled" \
        else cfg.singer_id.encoder_weights
    enc = VoiceEncoder(device="cpu", verbose=False, weights_fpath=weights_path)
    t0 = time.time()
    embed = _embed(args.input, enc, int(cfg.singer_id.emb_sr))
    print(f"[main] d-vector computed in {time.time() - t0:.2f}s")

    # -- Step 2: nearest singer by centroid -------------------------------
    if not os.path.exists(args.centroids):
        print(f"[main] missing centroid matrix {args.centroids}")
        return 1
    emb_all = np.load(args.centroids)
    singer_csv = args.singer_csv
    if not os.path.exists(singer_csv):
        print(f"[main] missing {singer_csv}")
        return 1
    import pandas as pd
    df = pd.read_csv(singer_csv)
    centroids = []
    singer_ids = []
    for s in sorted(df["singer_id"].unique()):
        mask = (df["singer_id"].values == s)
        c = emb_all[mask].mean(axis=0)
        c /= max(np.linalg.norm(c), 1e-12)
        centroids.append(c)
        singer_ids.append(int(s))
    centroids = np.stack(centroids, axis=0)
    sims = _cosine_sim_matrix(embed, centroids)
    best = int(np.argmax(sims))
    similarity = float(sims[best])
    print(f"[main] predicted singer: {singer_ids[best]}  "
          f"(cosine similarity = {similarity:.3f}  "
          f"n_wavs_in_cluster = {int((df['singer_id'] == singer_ids[best]).sum())})")

    # -- Step 3: MFCC + per-singer SVM verdict ----------------------------
    X = _mfcc_features(
        args.input,
        sampling_rate=int(cfg.data.sampling_rate),
        mfcc_order=int(cfg.data.mfcc_order),
        fft_size=int(cfg.data.fft_size),
        hop_length=int(cfg.data.hop_length),
        pre_emph=float(cfg.data.pre_emph),
        target_frames=int(cfg.data.train_frames),
    )

    verdicts = {}
    print(f"\n[main] clip: {os.path.basename(args.input)}")
    print(f"        under singer {singer_ids[best]}; X shape {X.shape}\n")
    for kern in ("linear", "rbf"):
        model_path = os.path.join(args.models_dir,
                                  f"singer_{singer_ids[best]}__{kern}.joblib")
        if not os.path.exists(model_path):
            print(f"        {kern}: model not found {model_path}")
            continue
        clf = joblib.load(model_path)
        pred = clf.predict(X).astype(int)
        chest_cnt = int((pred == -1).sum())
        falsetto_cnt = int((pred == 1).sum())
        pct_falsetto = falsetto_cnt / len(pred)
        verdict = "falsetto" if pct_falsetto > 0.5 else "chest"
        verdicts[kern] = verdict
        print(f"        {kern:7s}: chest={chest_cnt:2d}  "
              f"falsetto={falsetto_cnt:2d}  ({pct_falsetto * 100:.1f}%)  "
              f"-> {verdict}")

    # -- Summary ---------------------------------------------------------
    print()
    print(f"[main] clip-level final decision (majority over kernels):")
    vals = list(verdicts.values())
    if not vals:
        print("        n/a (no kernels available)")
    else:
        all_falsetto = sum(1 for v in vals if v == "falsetto")
        all_chest = sum(1 for v in vals if v == "chest")
        if all_falsetto == len(vals):
            print("        falsetto (both kernels)")
        elif all_chest == len(vals):
            print("        chest (both kernels)")
        else:
            print(f"        falsetto={all_falsetto}/{len(vals)}  chest={all_chest}/{len(vals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
