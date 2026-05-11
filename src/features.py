"""
Extraccion de features espectrales.

MFCC (Mel-Frequency Cepstral Coefficients): captura la envolvente espectral
de la senal en una escala perceptual (Mel) y la decorrela con DCT. Para
keyword spotting es preferible a Mel-Spectrogram porque:
  - Es mas compacto (40 vs 64+ canales).
  - Decorrelacionar las dimensiones reduce la redundancia que la CNN
    debe aprender a ignorar.
  - Las deltas (1ra y 2da derivada temporal) anaden informacion dinamica
    sin disparar la dimensionalidad.

La salida es un tensor (n_mfcc*3, n_frames) si include_deltas=True.
"""
from __future__ import annotations
import numpy as np
import librosa

from .config import (
    SAMPLE_RATE,
    N_MFCC,
    N_FFT,
    WIN_LENGTH,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
)


def compute_mfcc(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    include_deltas: bool = True,
) -> np.ndarray:
    """
    Calcula MFCC con la configuracion fijada en config.py.

    Pasos internos (los aplica librosa):
      1. STFT con ventana Hann de WIN_LENGTH muestras, hop HOP_LENGTH.
      2. Magnitud al cuadrado -> espectrograma de potencia.
      3. Filtro Mel con N_MELS bandas -> mel-spectrograma.
      4. log10 (en dB).
      5. DCT-II tipo 'ortho' -> coeficientes cepstrales.

    Devuelve matriz (n_mfcc o n_mfcc*3, n_frames) en float32.
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=False,
    )
    if not include_deltas:
        return mfcc.astype(np.float32)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    in_db: bool = True,
) -> np.ndarray:
    """Mel-spectrograma logaritmico. Util para visualizacion."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=n_mels,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    if in_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)


def standardize_per_feature(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Estandariza cada coeficiente MFCC (filas) restando su media y dividiendo
    entre su desviacion estandar dentro del mismo clip. Es invariante a
    ganancia global del microfono.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)
