"""
Entrenamiento del modelo base.

Lee data/processed/dataset.npz, entrena la CNN definida en src/model.py,
guarda el mejor modelo y exporta artefactos auxiliares:

    models/cnn_voz_base.keras           # mejor checkpoint (formato nativo TF)
    models/cnn_voz_base.tflite          # version TFLite float32 para inferencia rapida
    models/cnn_voz_base_int8.tflite     # version cuantizada INT8 (opcional)
    models/training_history.json        # loss/accuracy por epoca
    models/training_report.json         # metricas finales en test + reporte por clase
    models/confusion_matrix.npy         # matriz de confusion en test

Uso desde la raiz del proyecto:
    python -m src.train
    python -m src.train --epochs 80 --batch-size 32
    python -m src.train --no-class-weights --no-int8
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    PROCESSED_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    CLASSES,
    RANDOM_SEED,
)
from .model import build_cnn, compile_model, model_summary_string


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=False)
    return {
        "X_train": d["X_train"], "y_train": d["y_train"],
        "X_val": d["X_val"], "y_val": d["y_val"],
        "X_test": d["X_test"], "y_test": d["y_test"],
        "classes": d["classes"].tolist(),
    }


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    """
    sklearn.utils.class_weight.compute_class_weight('balanced', ...) puede
    fallar si una clase no aparece en y. Aqui hacemos el calculo manual
    equivalente y asignamos peso 1.0 a las clases ausentes.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = {}
    n_present = int((counts > 0).sum())
    for i, c in enumerate(counts):
        if c > 0:
            weights[i] = float(total / (n_present * c))
        else:
            weights[i] = 1.0
    return weights


def build_callbacks(
    ckpt_path: Path,
    patience_es: int,
    patience_lr: int,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def export_tflite_float(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)


def export_tflite_int8(
    model: tf.keras.Model,
    out_path: Path,
    representative_X: np.ndarray,
    n_samples: int = 200,
) -> None:
    """Cuantizacion INT8 con dataset representativo."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(representative_X), size=min(n_samples, len(representative_X)),
                     replace=False)

    def representative_dataset():
        for i in idx:
            yield [representative_X[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)


def evaluate_split(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    label: str,
) -> dict:
    """Evalua y devuelve un diccionario serializable con metricas."""
    if len(X) == 0:
        return {"split": label, "n": 0}
    probs = model.predict(X, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y).mean())
    p, r, f1, sup = precision_recall_fscore_support(
        y, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0, output_dict=True,
    )
    print(f"\n=== {label} ===  n={len(y)}  accuracy={acc:.4f}")
    print(classification_report(
        y, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    ))
    return {
        "split": label,
        "n": int(len(y)),
        "accuracy": acc,
        "per_class": {
            cls: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f1[i]), "support": int(sup[i])}
            for i, cls in enumerate(class_names)
        },
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la CNN base.")
    parser.add_argument("--dataset", type=Path, default=PROCESSED_DIR / "dataset.npz")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience-es", type=int, default=20,
                        help="Paciencia de EarlyStopping.")
    parser.add_argument("--patience-lr", type=int, default=7,
                        help="Paciencia de ReduceLROnPlateau.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-int8", action="store_true",
                        help="No exportar la version cuantizada INT8.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seeds(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Cargando dataset desde {args.dataset} ...")
    if not args.dataset.exists():
        raise SystemExit(
            f"No existe {args.dataset}. Corre primero:  python -m src.build_dataset"
        )
    data = load_dataset(args.dataset)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]
    print(f"      train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    print(f"      clases: {classes}")

    input_shape = X_train.shape[1:]
    print(f"\n[2/5] Construyendo modelo (input={input_shape}) ...")
    model = build_cnn(input_shape=input_shape, num_classes=NUM_CLASSES)
    model = compile_model(model, learning_rate=args.learning_rate)
    summary = model_summary_string(model)
    print(summary)
    (args.models_dir / "model_summary.txt").write_text(summary, encoding="utf-8")

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(y_train, NUM_CLASSES)
        print(f"\nPesos de clase: {class_weights}")

    ckpt_path = args.models_dir / "cnn_voz_base.keras"
    callbacks = build_callbacks(ckpt_path, args.patience_es, args.patience_lr)

    print(f"\n[3/5] Entrenando hasta {args.epochs} epocas (batch={args.batch_size}) ...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0
    print(f"      Entrenamiento finalizado en {elapsed:.1f}s")

    history_path = args.models_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  f, indent=2)
    print(f"      Historial guardado en {history_path}")

    print("\n[4/5] Evaluacion final ...")
    train_eval = evaluate_split(model, X_train, y_train, classes, "train")
    val_eval = evaluate_split(model, X_val, y_val, classes, "val")
    test_eval = evaluate_split(model, X_test, y_test, classes, "test")

    report = {
        "elapsed_seconds": elapsed,
        "epochs_run": len(history.history.get("loss", [])),
        "best_checkpoint": str(ckpt_path),
        "splits": {"train": train_eval, "val": val_eval, "test": test_eval},
        "class_weights": class_weights,
    }
    report_path = args.models_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if test_eval.get("confusion_matrix"):
        np.save(args.models_dir / "confusion_matrix.npy",
                np.array(test_eval["confusion_matrix"], dtype=np.int64))

    print("\n[5/5] Exportando TFLite ...")
    tflite_path = args.models_dir / "cnn_voz_base.tflite"
    export_tflite_float(model, tflite_path)
    print(f"      Float32 -> {tflite_path}  ({tflite_path.stat().st_size/1024:.1f} KB)")
    if not args.no_int8:
        try:
            int8_path = args.models_dir / "cnn_voz_base_int8.tflite"
            export_tflite_int8(model, int8_path, X_train)
            print(f"      INT8    -> {int8_path}  ({int8_path.stat().st_size/1024:.1f} KB)")
        except Exception as e:
            print(f"      [aviso] export INT8 fallo: {type(e).__name__}: {e}")

    print("\nListo.")


if __name__ == "__main__":
    main()
