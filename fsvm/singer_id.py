"""Singer-ID pseudo-labelling via ColorSplitter's Resemblyzer VoiceEncoder.

Pipeline:
  1. For every wav under <data.audio_dir>, compute a 256-d L2-normalised
     d-vector (VoiceEncoder).
  2. Cluster the 1280x256 matrix with the spectral-cluster rule shipped by
     ColorSplitter (3D-Speaker under Apache-2.0). We *force* the cluster
     count to N_SINGERS by passing `oracle_num=N_SINGERS` -- the CCMusic
     dataset README states there's 12 singers, but their identities are not
     published, forcing a curated N is the only honest path.
  3. Persist data/singer_id/embeddings.npy and data/singer_id/singer_id.csv.

Why this works for a "singer-dependent" SVM:
  - The paper trains one SVM per singer. We have no per-singer meta in the
    filenames, so we *recover* a 12-class singer identity from speaker
    timbre. Spectral clustering on Resemblyzer d-vectors breaks the corpus
    into 12 disjoint clusters which our downstream SVM treats as 12 singers
    (this is the best that the published dataset supports).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from resemblyzer import VoiceEncoder

# ColorSplitter's clustering utilities. Add the upstream path so we don't
# duplicate code.
CS_ROOT = r"H:/GitHub/ColorSplitter/ColorSplitter"
sys.path.insert(0, CS_ROOT)
from modules.cluster import SpectralCluster  # noqa: E402

from utils import ensure_dir, load_config, traverse_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    return p.parse_args()


def _read_wav(path: str):
    """Read wav as float32 numpy array (mono), preserving native sample rate.

    Resemblyzer expects a 1-D float32 ndarray via `embed_utterance`; the
    function will internally pad / chunk to its 1.6s partials window.
    """
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), int(sr)


def compute_embeddings(
    files: list[str],
    audio_root: str,
    encoder: VoiceEncoder,
    emb_sr: int,
) -> np.ndarray:
    """Return array of shape (len(files), 256) d-vectors."""
    out = np.zeros((len(files), 256), dtype=np.float32)
    t0 = time.time()
    for i, rel in enumerate(files):
        path = os.path.join(audio_root, rel)
        wav, nat_sr = _read_wav(path)
        # Resemblyzer is trained at 16 kHz. Re-sample if needed.
        if nat_sr != emb_sr:
            wav = librosa.resample(wav, orig_sr=nat_sr, target_sr=emb_sr)
        with torch.no_grad():
            embed = encoder.embed_utterance(wav)
        out[i] = embed
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(files)}  "
                  f"({(time.time() - t0):.1f}s)")
    return out


def cluster_embeddings(embeds: np.ndarray, n_speakers: int) -> np.ndarray:
    """Spectral-cluster with the cluster count *fixed* at n_speakers."""
    cl = SpectralCluster(
        min_num_spks=2, max_num_spks=14, pval=0.02, min_pnum=6,
        oracle_num=n_speakers,
    )
    return cl(embeds)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    audio_root = cfg.data.audio_dir
    files = traverse_dir(audio_root, extension="wav", is_sort=True, is_ext=True)

    cache_emb = cfg.singer_id.cache_emb
    cache_csv = cfg.singer_id.cache_csv
    ensure_dir(os.path.dirname(cache_emb))
    ensure_dir(os.path.dirname(cache_csv))

    if os.path.exists(cache_emb) and os.path.exists(cache_csv):
        print(f"[singer_id] cached embeddings+csv already present -> reusing")
        return 0

    weights = cfg.singer_id.encoder_weights
    if weights == "bundled":
        # Use resemblyzer's own pretrained (default behaviour when weights_fpath is None)
        weights_fpath = None  # type: ignore[assignment]
        print("[singer_id] using bundled resemblyzer pretrained.pt")
    elif weights == "colorsplitter":
        weights_fpath = (
            r"H:/GitHub/ColorSplitter/ColorSplitter/pretrain/encoder_1570000.bak"
        )
        weights = weights_fpath  # just to keep the log consistent
        print(f"[singer_id] using ColorSplitter weights {weights_fpath}")
    else:
        weights_fpath = weights  # treat as a direct path
        print(f"[singer_id] using custom weights path {weights_fpath}")
    emb_sr = cfg.singer_id.emb_sr

    print(f"[singer_id] loading VoiceEncoder from {weights}")
    enc = VoiceEncoder(device="cpu", verbose=False, weights_fpath=weights_fpath)

    print(f"[singer_id] embedding {len(files)} wavs at {emb_sr} Hz ...")
    embeds = compute_embeddings(files, audio_root, enc, emb_sr)
    np.save(cache_emb, embeds)
    print(f"[singer_id] embeddings saved -> {cache_emb}  shape={embeds.shape}")

    n_speakers = int(cfg.singer_id.n_speakers)
    print(f"[singer_id] spectral clustering, oracle_num={n_speakers}")
    labels = cluster_embeddings(embeds, n_speakers=n_speakers)
    print(f"[singer_id] got {len(set(labels))} clusters  sizes={np.bincount(labels).tolist()}")

    df = pd.DataFrame({
        "filename": [f.replace(".wav", "") for f in files],
        "singer_id": labels.astype(int),
    })
    df.to_csv(cache_csv, index=False)
    print(f"[singer_id] csv saved -> {cache_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
