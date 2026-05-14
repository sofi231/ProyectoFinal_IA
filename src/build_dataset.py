"""
Pipeline principal de construccion del dataset.

Recorre Audios/, aplica preprocesamiento, calcula MFCC, hace split
estratificado y guarda un .npz con X_train, y_train, X_val, y_val,
X_test, y_test, label_map y metadata.

Uso desde la raiz del proyecto:
    python -m src.build_dataset
    python -m src.build_dataset --no-augment
    python -m src.build_dataset --aug-per-sample 5
"""
from __future__ import annotations
import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    AUDIO_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
    CLIP_SAMPLES,
    LABEL_TO_INDEX,
    CLASSES,
    VAL_FRACTION,
    TEST_FRACTION,
    RANDOM_SEED,
    AUG_PER_SAMPLE,
)
from .audio_utils import preprocess_clip
from .features import compute_mfcc, standardize_per_feature
from .augmentation import augment_waveform, spec_augment


FILENAME_RE = re.compile(
    r"^(?P<label>[A-Z]+)_(?P<speaker>.+?)_(?P<idx>\d+)_(?P<env>silencio|ruido|ambiente|habla)\.wav$",
    re.IGNORECASE,
)


@dataclass
class Sample:
    path: Path
    label: str
    speaker: str
    env: str
    idx: int


def parse_filename(p: Path) -> Sample | None:
    m = FILENAME_RE.match(p.name)
    if not m:
        return None
    label = m.group("label").upper()
    if label not in LABEL_TO_INDEX:
        return None
    return Sample(
        path=p,
        label=label,
        speaker=m.group("speaker").lower(),
        env=m.group("env").lower(),
        idx=int(m.group("idx")),
    )


def discover_samples(audio_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    skipped: list[str] = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        s = parse_filename(wav)
        if s is None:
            skipped.append(wav.name)
        else:
            samples.append(s)
    if skipped:
        print(f"[aviso] {len(skipped)} archivos con nombre invalido fueron ignorados:")
        for n in skipped[:5]:
            print(f"   - {n}")
        if len(skipped) > 5:
            print(f"   ... (+{len(skipped) - 5} mas)")
    return samples


def stratified_split(
    samples: list[Sample],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    labels = [s.label for s in samples]
    train_val, test = train_test_split(
        samples,
        test_size=test_fraction,
        random_state=seed,
        stratify=labels,
    )
    rel_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=rel_val,
        random_state=seed,
        stratify=[s.label for s in train_val],
    )
    return train, val, test


def sample_to_features(
    s: Sample,
    apply_vad: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (waveform_fija_1.5s, mfcc_estandarizada)."""
    y = preprocess_clip(s.path, sr=SAMPLE_RATE, apply_vad=apply_vad)
    mfcc = compute_mfcc(y, sr=SAMPLE_RATE)
    mfcc = standardize_per_feature(mfcc)
    return y, mfcc


def build_split(
    samples: list[Sample],
    augment: bool,
    aug_per_sample: int,
    rng: np.random.Generator,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X: list[np.ndarray] = []
    y_idx: list[int] = []
    meta: list[dict] = []

    for s in tqdm(samples, desc=desc):
        wav, mfcc = sample_to_features(s, apply_vad=True)
        X.append(mfcc)
        y_idx.append(LABEL_TO_INDEX[s.label])
        meta.append({"file": s.path.name, "label": s.label, "speaker": s.speaker,
                     "env": s.env, "aug": "none"})

        if not augment:
            continue
        for k in range(aug_per_sample):
            wav_aug = augment_waveform(wav, rng, sr=SAMPLE_RATE)
            mfcc_aug = compute_mfcc(wav_aug, sr=SAMPLE_RATE)
            mfcc_aug = standardize_per_feature(mfcc_aug)
            if rng.random() < 0.5:
                mfcc_aug = spec_augment(mfcc_aug, rng)
            X.append(mfcc_aug)
            y_idx.append(LABEL_TO_INDEX[s.label])
            meta.append({"file": s.path.name, "label": s.label, "speaker": s.speaker,
                         "env": s.env, "aug": f"v{k+1}"})

    X_arr = np.stack(X, axis=0).astype(np.float32)
    y_arr = np.asarray(y_idx, dtype=np.int64)
    return X_arr, y_arr, meta


def summarize(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"\n[{name}] X shape={X.shape}  y shape={y.shape}")
    print(f"   dtype={X.dtype}  min={X.min():.3f}  max={X.max():.3f}  mean={X.mean():.3f}")
    bincount = np.bincount(y, minlength=len(CLASSES))
    for cls, c in zip(CLASSES, bincount):
        print(f"     {cls:14s} {c}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el dataset de MFCC.")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR,
                        help="Directorio raiz con subcarpetas por clase.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "dataset.npz",
                        help="Ruta del .npz a generar.")
    parser.add_argument("--no-augment", action="store_true",
                        help="Desactiva data augmentation en train.")
    parser.add_argument("--aug-per-sample", type=int, default=AUG_PER_SAMPLE,
                        help="Numero de copias aumentadas por muestra de train.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Descubriendo audios en {args.audio_dir} ...")
    samples = discover_samples(args.audio_dir)
    if not samples:
        raise SystemExit("No se encontraron audios validos. Revisa el formato del nombre.")
    print(f"      {len(samples)} muestras encontradas.")

    counts = {c: 0 for c in CLASSES}
    for s in samples:
        counts[s.label] += 1
    print("      Distribucion por clase:")
    for c, n in counts.items():
        flag = "" if n > 0 else "  <-- pendiente de grabar"
        print(f"        {c:14s} {n}{flag}")

    print(f"\n[2/4] Split estratificado (val={VAL_FRACTION:.0%}, test={TEST_FRACTION:.0%}) ...")
    classes_with_data = {c for c, n in counts.items() if n >= 3}
    usable = [s for s in samples if s.label in classes_with_data]
    if len(usable) < len(samples):
        skipped = len(samples) - len(usable)
        print(f"      [aviso] {skipped} muestras se omiten porque su clase tiene <3 ejemplos.")
    train, val, test = stratified_split(usable, VAL_FRACTION, TEST_FRACTION, args.seed)
    print(f"      train={len(train)}  val={len(val)}  test={len(test)}")

    rng = np.random.default_rng(args.seed)

    print("\n[3/4] Extrayendo features ...")
    X_train, y_train, meta_train = build_split(
        train, augment=not args.no_augment, aug_per_sample=args.aug_per_sample,
        rng=rng, desc="train",
    )
    X_val, y_val, meta_val = build_split(
        val, augment=False, aug_per_sample=0, rng=rng, desc="val",
    )
    X_test, y_test, meta_test = build_split(
        test, augment=False, aug_per_sample=0, rng=rng, desc="test",
    )

    summarize("train", X_train, y_train)
    summarize("val", X_val, y_val)
    summarize("test", X_test, y_test)

    print(f"\n[4/4] Guardando en {args.output} ...")
    np.savez_compressed(
        args.output,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=np.array(CLASSES),
        sample_rate=np.int32(SAMPLE_RATE),
        clip_samples=np.int32(CLIP_SAMPLES),
    )

    meta_path = args.output.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"train": meta_train, "val": meta_val, "test": meta_test}, f, indent=2)
    print(f"      Metadata en {meta_path}")
    print("\nListo.")


if __name__ == "__main__":
    main()
