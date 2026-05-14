"""
Inferencia en tiempo real desde microfono.

Pipeline:
    Microfono (sounddevice) -> ring buffer
        -> VAD por energia RMS (deteccion de borde subida/bajada)
        -> Segmento aislado, padding/recorte a 1.5 s
        -> MFCC + estandarizacion (igual que entrenamiento)
        -> Inferencia con TFLite (rapida) o Keras (.keras)
        -> Decision: ejecutar accion o ignorar (clase RUIDO / baja confianza)

Mide y reporta:
    - latencia de extraccion de features (ms)
    - latencia de inferencia (ms)
    - latencia total (captura -> resultado)
    - confianza (max softmax)

Uso:
    python -m src.realtime
    python -m src.realtime --model models/cnn_voz_base.tflite
    python -m src.realtime --model models/cnn_voz_base.keras
    python -m src.realtime --device 1 --threshold 0.02
    python -m src.realtime --list-devices
"""
from __future__ import annotations
import argparse
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import (
    SAMPLE_RATE,
    CLIP_SAMPLES,
    CLASSES,
    INDEX_TO_LABEL,
    MODELS_DIR,
    VAD_FRAME_MS,
    VAD_HOP_MS,
)
from .audio_utils import peak_normalize, fix_length
from .features import compute_mfcc, standardize_per_feature


# ---------- Backend de inferencia ------------------------------------------

class InferenceBackend:
    """Envoltorio comun para .keras o .tflite. La interfaz es predict(x) -> probs."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.is_tflite = model_path.suffix == ".tflite"
        self._load()

    def _load(self) -> None:
        if self.is_tflite:
            import tensorflow as tf
            self.interp = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interp.allocate_tensors()
            self.in_det = self.interp.get_input_details()[0]
            self.out_det = self.interp.get_output_details()[0]
            self.input_dtype = self.in_det["dtype"]
            self.input_quant = self.in_det.get("quantization_parameters", None)
            self.output_dtype = self.out_det["dtype"]
            self.output_quant = self.out_det.get("quantization_parameters", None)
        else:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path))

    def predict(self, x_2d: np.ndarray) -> np.ndarray:
        """x_2d : (n_features, n_frames) -> probs (n_classes,)."""
        x = x_2d[None, ...]  # (1, n_features, n_frames)
        if not self.is_tflite:
            return self.model.predict(x, verbose=0)[0]

        # Manejo de cuantizacion INT8 si aplica
        if self.input_dtype == np.int8 and self.input_quant is not None:
            scale = self.input_quant["scales"][0]
            zp = self.input_quant["zero_points"][0]
            x_q = np.round(x / scale + zp).astype(np.int8)
            self.interp.set_tensor(self.in_det["index"], x_q)
        else:
            self.interp.set_tensor(self.in_det["index"], x.astype(np.float32))

        self.interp.invoke()
        out = self.interp.get_tensor(self.out_det["index"])[0]

        if self.output_dtype == np.int8 and self.output_quant is not None:
            scale = self.output_quant["scales"][0]
            zp = self.output_quant["zero_points"][0]
            out = (out.astype(np.float32) - zp) * scale
        else:
            out = out.astype(np.float32)
        return out


# ---------- Captura + VAD --------------------------------------------------

class RealtimeRecognizer:
    """
    Bucle de captura/inferencia. Maneja un ring buffer en samples crudos y
    un estado VAD basado en RMS por frame.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        sr: int = SAMPLE_RATE,
        device: int | None = None,
        block_ms: int = 32,
        frame_ms: int = VAD_FRAME_MS,
        hop_ms: int = VAD_HOP_MS,
        min_speech_ms: int = 200,
        end_silence_ms: int = 400,
        pre_roll_ms: int = 200,
        max_segment_ms: int = 2000,
        threshold: float | None = None,
        threshold_factor: float = 4.0,
        min_threshold: float = 5e-3,
        confidence_min: float = 0.50,
    ):
        self.backend = backend
        self.sr = sr
        self.device = device
        self.block_samples = int(sr * block_ms / 1000)
        self.frame_len = int(sr * frame_ms / 1000)
        self.hop_len = int(sr * hop_ms / 1000)
        self.min_speech_frames = max(1, int(min_speech_ms / hop_ms))
        self.end_silence_frames = max(1, int(end_silence_ms / hop_ms))
        self.pre_roll_samples = int(sr * pre_roll_ms / 1000)
        self.max_segment_samples = int(sr * max_segment_ms / 1000)
        self.threshold = threshold
        self.threshold_factor = threshold_factor
        self.min_threshold = min_threshold
        self.confidence_min = confidence_min

        # Estado
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._tail = np.zeros(self.frame_len, dtype=np.float32)
        self._capture_start_ts: float | None = None
        self._reset_segment()

    def _reset_segment(self) -> None:
        self._pre_roll = deque(maxlen=self.pre_roll_samples)
        self._segment: list[np.ndarray] = []
        self._segment_len = 0
        self._active = False
        self._silence_count = 0
        self._speech_count = 0
        self._capture_start_ts = None

    # --- Calibracion ---
    def calibrate(self, seconds: float = 1.0) -> float:
        """Captura ruido de fondo y devuelve un umbral RMS recomendado."""
        if self.threshold is not None:
            print(f"[calibracion] umbral fijo configurado: {self.threshold:.4f}")
            return self.threshold
        print(f"[calibracion] mantenga silencio durante {seconds:.1f} s ...")
        rec = sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1,
                     dtype="float32", device=self.device)
        sd.wait()
        y = rec.flatten()
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        thr = max(rms * self.threshold_factor, self.min_threshold)
        print(f"[calibracion] ruido RMS={rms:.5f}  umbral={thr:.5f}")
        self.threshold = thr
        return thr

    # --- Callback de audio ---
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[sd warning] {status}", file=sys.stderr)
        # mono float32
        block = indata[:, 0].astype(np.float32).copy()
        self._q.put(block)

    # --- VAD por frame ---
    def _iter_frame_rms(self, buf: np.ndarray):
        """Genera (rms, end_index_in_buf) por frame (con overlap)."""
        n = len(buf)
        i = 0
        while i + self.frame_len <= n:
            seg = buf[i : i + self.frame_len]
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12))
            yield rms, i + self.frame_len
            i += self.hop_len

    # --- Pipeline de inferencia sobre un segmento ---
    def _infer(self, segment: np.ndarray) -> dict:
        t0 = time.perf_counter()
        y = peak_normalize(segment)
        y = fix_length(y, target_len=CLIP_SAMPLES)
        mfcc = compute_mfcc(y, sr=self.sr)
        mfcc = standardize_per_feature(mfcc)
        t_feat = time.perf_counter()
        probs = self.backend.predict(mfcc)
        t_inf = time.perf_counter()

        idx = int(np.argmax(probs))
        return {
            "label": INDEX_TO_LABEL[idx],
            "confidence": float(probs[idx]),
            "probs": probs,
            "feat_ms": (t_feat - t0) * 1000,
            "inf_ms": (t_inf - t_feat) * 1000,
            "total_ms": (t_inf - t0) * 1000,
            "n_samples": int(len(segment)),
        }

    # --- Bucle principal ---
    def run(self) -> None:
        self.calibrate()
        print()
        print(f"Escuchando en device={self.device}  sr={self.sr} Hz  umbral={self.threshold:.5f}")
        print("Habla un comando. Ctrl+C para salir.\n")

        with sd.InputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=self.block_samples,
            callback=self._callback,
        ):
            buf = np.zeros(0, dtype=np.float32)
            try:
                while True:
                    block = self._q.get()
                    buf = np.concatenate([buf, block])
                    if len(buf) < self.frame_len:
                        continue

                    consumed_up_to = 0
                    for rms, end in self._iter_frame_rms(buf):
                        is_speech = rms > self.threshold

                        if not self._active:
                            # IDLE: mantener pre-roll
                            seg = buf[max(0, end - self.hop_len) : end]
                            for s in seg:
                                self._pre_roll.append(s)
                            if is_speech:
                                self._speech_count += 1
                                if self._speech_count >= self.min_speech_frames:
                                    # transicion a ACTIVO
                                    self._active = True
                                    self._capture_start_ts = time.perf_counter()
                                    pre = np.fromiter(self._pre_roll, dtype=np.float32)
                                    self._segment = [pre]
                                    self._segment_len = len(pre)
                                    self._silence_count = 0
                            else:
                                self._speech_count = 0
                        else:
                            # ACTIVO: acumular
                            seg = buf[max(0, end - self.hop_len) : end]
                            self._segment.append(seg)
                            self._segment_len += len(seg)
                            if is_speech:
                                self._silence_count = 0
                            else:
                                self._silence_count += 1

                            if (self._silence_count >= self.end_silence_frames
                                    or self._segment_len >= self.max_segment_samples):
                                seg_arr = np.concatenate(self._segment)
                                t_capture = (time.perf_counter() - self._capture_start_ts) * 1000 \
                                    if self._capture_start_ts else 0.0
                                self._emit(seg_arr, t_capture)
                                self._reset_segment()

                        consumed_up_to = end

                    # mantener overlap: dejar los ultimos frame_len samples
                    if consumed_up_to > self.frame_len:
                        buf = buf[consumed_up_to - self.frame_len:]
            except KeyboardInterrupt:
                print("\nDetenido por el usuario.")

    # --- Reporte por deteccion ---
    def _emit(self, segment: np.ndarray, capture_ms: float) -> None:
        out = self._infer(segment)
        bar = "*" * int(out["confidence"] * 20)
        action = self._decide_action(out["label"], out["confidence"])
        print(
            f"  -> {out['label']:11s}  conf={out['confidence']:.2f} {bar:20s}  "
            f"dur={out['n_samples']/self.sr*1000:.0f}ms  "
            f"feat={out['feat_ms']:.1f}ms  inf={out['inf_ms']:.1f}ms  "
            f"total={out['total_ms']:.1f}ms  {action}"
        )

    def _decide_action(self, label: str, conf: float) -> str:
        if conf < self.confidence_min:
            return "[ignorado: baja confianza]"
        if label == "RUIDO":
            return "[ignorado: rechazo]"
        return f"[ACCION: {label}]"


# ---------- CLI ------------------------------------------------------------

def list_devices() -> None:
    devs = sd.query_devices()
    print("Dispositivos de entrada:")
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            marker = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}  ch={d['max_input_channels']}  "
                  f"sr_default={int(d['default_samplerate'])} Hz{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real.")
    parser.add_argument("--model", type=Path,
                        default=MODELS_DIR / "cnn_voz_base.tflite",
                        help="Ruta al .tflite o .keras a usar.")
    parser.add_argument("--device", type=int, default=None,
                        help="Indice del dispositivo de entrada (sounddevice).")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral RMS para VAD (omitir = auto-calibrar).")
    parser.add_argument("--threshold-factor", type=float, default=4.0,
                        help="Factor multiplicativo sobre el ruido de fondo.")
    parser.add_argument("--confidence-min", type=float, default=0.50,
                        help="Confianza minima para ejecutar accion.")
    parser.add_argument("--min-speech-ms", type=int, default=200)
    parser.add_argument("--end-silence-ms", type=int, default=400)
    parser.add_argument("--pre-roll-ms", type=int, default=200)
    parser.add_argument("--max-segment-ms", type=int, default=2000)
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not args.model.exists():
        raise SystemExit(f"No existe el modelo: {args.model}.\n"
                         f"Corre primero:  python -m src.train")

    print(f"Cargando modelo {args.model.name} ...")
    backend = InferenceBackend(args.model)
    print(f"  backend: {'TFLite' if backend.is_tflite else 'Keras'}")
    if backend.is_tflite:
        print(f"  input dtype: {backend.input_dtype.__name__}  "
              f"output dtype: {backend.output_dtype.__name__}")

    rec = RealtimeRecognizer(
        backend=backend,
        device=args.device,
        threshold=args.threshold,
        threshold_factor=args.threshold_factor,
        confidence_min=args.confidence_min,
        min_speech_ms=args.min_speech_ms,
        end_silence_ms=args.end_silence_ms,
        pre_roll_ms=args.pre_roll_ms,
        max_segment_ms=args.max_segment_ms,
    )
    rec.run()


if __name__ == "__main__":
    main()
