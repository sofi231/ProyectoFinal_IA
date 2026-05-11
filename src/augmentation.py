"""
Tecnicas de aumento de datos para audio.

Implementa cuatro tecnicas exigidas/recomendadas por el PDF:
  1. Time shifting: desplaza la senal en el eje temporal y rellena con cero.
  2. Pitch shifting: cambia el tono sin alterar la duracion.
  3. Inyeccion de ruido gaussiano calibrada por SNR.
  4. SpecAugment: enmascara bloques de tiempo y/o frecuencia sobre el MFCC.

Las tres primeras operan sobre la senal cruda (1D); SpecAugment opera
sobre el tensor de features (2D). Cada funcion es determinista respecto
a un np.random.Generator pasado como argumento para que el dataset sea
reproducible.
"""
from __future__ import annotations
import numpy as np
import librosa

from .config import (
    SAMPLE_RATE,
    TIME_SHIFT_MAX_MS,
    PITCH_SHIFT_SEMITONES_RANGE,
    NOISE_SNR_DB_RANGE,
    SPECAUG_TIME_MASK_MAX,
    SPECAUG_FREQ_MASK_MAX,
    SPECAUG_N_MASKS,
)


def time_shift(
    y: np.ndarray,
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    max_ms: int = TIME_SHIFT_MAX_MS,
) -> np.ndarray:
    """Desplaza la senal hasta +/- max_ms milisegundos, rellena con ceros."""
    max_shift = int(sr * max_ms / 1000)
    if max_shift <= 0 or len(y) == 0:
        return y.copy()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return y.copy()
    if shift > 0:
        out = np.concatenate([np.zeros(shift, dtype=y.dtype), y[:-shift]])
    else:
        out = np.concatenate([y[-shift:], np.zeros(-shift, dtype=y.dtype)])
    return out.astype(np.float32)


def pitch_shift(
    y: np.ndarray,
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
    semitones_range: tuple[float, float] = PITCH_SHIFT_SEMITONES_RANGE,
) -> np.ndarray:
    """Cambia el tono en un numero aleatorio de semitonos dentro del rango."""
    n_steps = float(rng.uniform(*semitones_range))
    if abs(n_steps) < 1e-3:
        return y.copy()
    out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return out.astype(np.float32)


def add_gaussian_noise(
    y: np.ndarray,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = NOISE_SNR_DB_RANGE,
) -> np.ndarray:
    """
    Inyecta ruido gaussiano con SNR aleatoria dentro del rango.
    SNR = 10*log10(P_signal / P_noise) => P_noise = P_signal / 10^(SNR/10).
    """
    snr_db = float(rng.uniform(*snr_db_range))
    p_signal = float(np.mean(y.astype(np.float64) ** 2))
    if p_signal <= 1e-12:
        return y.copy()
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)


def spec_augment(
    spec: np.ndarray,
    rng: np.random.Generator,
    time_mask_max: int = SPECAUG_TIME_MASK_MAX,
    freq_mask_max: int = SPECAUG_FREQ_MASK_MAX,
    n_masks: int = SPECAUG_N_MASKS,
    mask_value: float | None = None,
) -> np.ndarray:
    """
    Aplica n_masks mascaras de tiempo y n_masks de frecuencia sobre spec
    (forma (n_features, n_frames)). El valor de relleno es la media del
    propio espectrograma (equivalente a 'cero' tras estandarizar).
    """
    out = spec.copy()
    n_feat, n_frames = out.shape
    fill = float(out.mean()) if mask_value is None else float(mask_value)

    for _ in range(n_masks):
        if n_frames > 1 and time_mask_max > 0:
            t = int(rng.integers(0, min(time_mask_max, n_frames) + 1))
            if t > 0:
                t0 = int(rng.integers(0, n_frames - t + 1))
                out[:, t0 : t0 + t] = fill
        if n_feat > 1 and freq_mask_max > 0:
            f = int(rng.integers(0, min(freq_mask_max, n_feat) + 1))
            if f > 0:
                f0 = int(rng.integers(0, n_feat - f + 1))
                out[f0 : f0 + f, :] = fill
    return out.astype(np.float32)


def augment_waveform(
    y: np.ndarray,
    rng: np.random.Generator,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Aplica una composicion aleatoria de tecnicas en el dominio del tiempo.
    Cada tecnica se activa con probabilidad 0.5; si ninguna se activa,
    forzamos al menos time_shift para garantizar variabilidad.
    """
    applied = False
    out = y
    if rng.random() < 0.5:
        out = time_shift(out, rng, sr=sr)
        applied = True
    if rng.random() < 0.5:
        out = pitch_shift(out, rng, sr=sr)
        applied = True
    if rng.random() < 0.5:
        out = add_gaussian_noise(out, rng)
        applied = True
    if not applied:
        out = time_shift(out, rng, sr=sr)
    return out.astype(np.float32)
