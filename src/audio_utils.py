"""
Utilidades de carga y pre-procesamiento de audio.

Funciones principales:
- load_wav: lee un WAV, fuerza mono y resamplea a SAMPLE_RATE.
- peak_normalize: normaliza la amplitud de la senal a [-1, 1].
- trim_by_energy: VAD simple por energia RMS por frame.
- fix_length: hace pad/truncado a CLIP_SAMPLES (zero-pad centrado).
- preprocess_clip: pipeline completo carga -> normaliza -> VAD -> fix_length.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa

from .config import (
    SAMPLE_RATE,
    CLIP_SAMPLES,
    VAD_FRAME_MS,
    VAD_HOP_MS,
    VAD_ENERGY_PERCENTILE,
    VAD_PAD_MS,
)


def load_wav(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Carga un WAV como mono float32 y lo resamplea a sr."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def peak_normalize(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Escala la senal para que su pico absoluto sea 1.0."""
    peak = float(np.max(np.abs(y)))
    if peak < eps:
        return y
    return (y / peak).astype(np.float32)


def _frame_energies(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """RMS por frame. Implementacion sencilla sin librosa para que sea facil de explicar."""
    n_frames = 1 + max(0, (len(y) - frame_len) // hop_len)
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = y[i * hop_len : i * hop_len + frame_len]
        energies[i] = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
    return energies


def trim_by_energy(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = VAD_FRAME_MS,
    hop_ms: int = VAD_HOP_MS,
    energy_percentile: float = VAD_ENERGY_PERCENTILE,
    pad_ms: int = VAD_PAD_MS,
) -> np.ndarray:
    """
    VAD por energia: marca como 'voz' los frames cuya RMS supere un umbral
    derivado del percentil energy_percentile de la distribucion de RMS
    del propio clip. Recorta antes del primer frame de voz y despues del
    ultimo, anadiendo un margen de pad_ms para no cortar fonemas.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if len(y) < frame_len:
        return y

    energies = _frame_energies(y, frame_len, hop_len)
    threshold = np.percentile(energies, energy_percentile)
    threshold = max(threshold, energies.max() * 0.05)

    voiced = np.where(energies > threshold)[0]
    if voiced.size == 0:
        return y

    pad_samples = int(sr * pad_ms / 1000)
    start = max(0, voiced[0] * hop_len - pad_samples)
    end = min(len(y), voiced[-1] * hop_len + frame_len + pad_samples)
    return y[start:end]


def fix_length(y: np.ndarray, target_len: int = CLIP_SAMPLES) -> np.ndarray:
    """
    Ajusta y a target_len muestras:
      - si es mas corto: zero-pad centrado.
      - si es mas largo: recorta centrado.
    """
    n = len(y)
    if n == target_len:
        return y.astype(np.float32)
    if n < target_len:
        pad_total = target_len - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)
    offset = (n - target_len) // 2
    return y[offset : offset + target_len].astype(np.float32)


def preprocess_clip(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    apply_vad: bool = True,
) -> np.ndarray:
    """Pipeline completo: carga -> normaliza -> VAD (opcional) -> fix_length."""
    y = load_wav(path, sr=sr)
    y = peak_normalize(y)
    if apply_vad:
        y = trim_by_energy(y, sr=sr)
        y = peak_normalize(y)
    y = fix_length(y, target_len=CLIP_SAMPLES)
    return y
