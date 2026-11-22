"""Versi lokal `dinov2_features.py` ReStraV (upstream commit c51ee52).

Perbedaan dengan upstream (rincian di PERUBAHAN.md):
  1. Decoder: PyAV menggantikan torchcodec (torchcodec di Windows butuh FFmpeg shared).
     Aturan sampling dibuat sama: T frame berjarak window_sec/(T-1) detik mulai dari
     max(0, tengah - window/2); frame yang ditampilkan pada waktu t dipilih; kalau
     waktu melewati akhir video, frame terakhir diulang ("repeat_last").
  2. DINOv2 dijalankan per video (24 frame), bukan 128 video sekaligus, agar muat di 4 GB VRAM.
  3. Model DINOv2 dimuat saat dibutuhkan (bukan saat import).
  4. Selain Z lengkap (CLS+patch), CLS token per frame dikembalikan terpisah untuk disimpan.

Fungsi geometri (compute_temporal_geometry, moment4, features_from_Z) disalin apa adanya.
"""
import warnings

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore", message="xFormers is available.*", category=UserWarning)

_DINOV2 = None


def load_dinov2(device):
    global _DINOV2
    if _DINOV2 is None:
        _DINOV2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()
    return _DINOV2.to(device)


# ---------------------------------------------------------------- decoding
def _frame_time(frame, stream):
    if frame.pts is None:
        return float(frame.time or 0.0)
    return float(frame.pts * stream.time_base)


def video_duration(container, stream):
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0


def decode_clip(video_path, T=24, window_sec=2.0, center_time=None):
    """Kembalikan tensor uint8 [T, 3, H, W], mengikuti aturan sampling upstream."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        dur = video_duration(container, stream)
        if dur <= 0:
            raise ValueError(f"Could not read video duration: {video_path}")
        if center_time is None:
            center_time = dur / 2.0
        half = window_sec / 2
        start = max(0.0, center_time - half)
        delta = window_sec / max(T - 1, 1)
        targets = [start + i * delta for i in range(T)]

        # seek sedikit sebelum start, lalu decode berurutan
        if start > 0 and stream.time_base is not None:
            container.seek(int(max(0.0, start - 1.0) / stream.time_base), stream=stream,
                           backward=True, any_frame=False)

        frames = []  # (waktu, ndarray RGB)
        last_needed = targets[-1]
        for frame in container.decode(stream):
            t = _frame_time(frame, stream)
            frames.append((t, frame.to_ndarray(format="rgb24")))
            if t > last_needed:
                break
            # buang frame jauh sebelum start supaya hemat memori
            if len(frames) > 2 and frames[1][0] <= start:
                frames.pop(0)

    if not frames:
        raise ValueError(f"No frames decoded: {video_path}")

    times = np.array([t for t, _ in frames])
    out = []
    for t in targets:
        idx = int(np.searchsorted(times, t + 1e-6, side="right") - 1)
        idx = min(max(idx, 0), len(frames) - 1)  # sebelum frame pertama -> pertama; lewat akhir -> terakhir
        out.append(frames[idx][1])
    clip = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).contiguous()  # [T,3,H,W]
    return clip


def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([TF.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True) for x in frames])


# ---------------------------------------------------------------- embedding
@torch.no_grad()
def embed_video(video_path, device, T=24, window_sec=2.0):
    """Kembalikan (Z [1, T, 257*384], cls [T, 384]) untuk satu video."""
    model = load_dinov2(device)
    clip = decode_clip(video_path, T=T, window_sec=window_sec)
    batch = preprocess(clip).to(device)
    feats = model.forward_features(batch)
    cls = feats["x_norm_clstoken"]                       # [T, 384]
    patches = feats["x_norm_patchtokens"]                # [T, 256, 384]
    tokens = torch.cat([cls.unsqueeze(1), patches], dim=1)
    Z = tokens.flatten(1).unsqueeze(0)                   # [1, T, 98688]
    return Z, cls


# ---------------------------------------------------------------- geometri (salinan upstream)
def compute_temporal_geometry(Z):
    delta = Z[:, 1:, :] - Z[:, :-1, :]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return d, theta


def moment4(x):
    mu = x.mean(dim=-1)
    mn = x.amin(dim=-1)
    mx = x.amax(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    return mu, mn, mx, var


def features_from_Z(Z):
    d, t = compute_temporal_geometry(Z)
    d7 = d[:, :7]
    t6 = t[:, :6]
    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1)
    return torch.cat([d7, t6, stats], dim=1)
