"""
Arquitectura del modelo base: CNN 2D sobre MFCC.

Entrada esperada: tensor (n_features, n_frames) por muestra, p. ej. (120, 151)
para MFCC(40) + delta + delta-delta sobre 1.5 s a hop 10 ms. El modelo
agrega internamente la dimension de canal -> (n_features, n_frames, 1).

Justificacion de la arquitectura:
- Conv2D trata la matriz MFCC como una imagen tiempo x frecuencia.
- BatchNormalization estabiliza el entrenamiento con dataset pequeno.
- MaxPooling reduce dimensionalidad y aporta invariancia local a pequenas
  traslaciones (timbre/pronunciacion).
- Dropout incremental (0.2 -> 0.3 -> 0.4 -> 0.5) actua como regularizador.
- GlobalAveragePooling2D (en lugar de Flatten) reduce drasticamente el
  numero de parametros de la cabeza densa y disminuye el riesgo de
  overfitting en corpus reducidos.
- Cabeza densa pequena (64 -> NUM_CLASSES) con softmax.

El modelo es suficientemente compacto para que, en una segunda iteracion,
pueda exportarse a TFLite (incluso TFLite Micro tras cuantizacion INT8).
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from .config import NUM_CLASSES


def build_cnn(
    input_shape: tuple[int, int],
    num_classes: int = NUM_CLASSES,
    base_filters: int = 32,
    dropout_rates: tuple[float, float, float, float] = (0.15, 0.25, 0.35, 0.45),
    l2_reg: float = 1e-4,
    name: str = "cnn_voz_base",
) -> tf.keras.Model:
    """
    Construye la CNN base.

    Parametros
    ----------
    input_shape : (n_features, n_frames). El canal se agrega adentro.
    num_classes : numero de clases de salida.
    base_filters : filtros del primer bloque; los siguientes duplican.
    dropout_rates : (block1, block2, block3, head).
    l2_reg : regularizacion L2 de los kernels conv.

    Retorna
    -------
    tf.keras.Model con softmax en la salida.
    """
    reg = regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None

    inputs = layers.Input(shape=input_shape, name="mfcc_input")
    x = layers.Reshape((*input_shape, 1), name="add_channel")(inputs)

    # Bloque 1
    x = layers.Conv2D(base_filters, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rates[0], name="drop1")(x)

    # Bloque 2
    x = layers.Conv2D(base_filters * 2, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rates[1], name="drop2")(x)

    # Bloque 3
    x = layers.Conv2D(base_filters * 2, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rates[2], name="drop3")(x)

    # Cabeza
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.ReLU(name="relu_head")(x)
    x = layers.Dropout(dropout_rates[3], name="drop_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Compila con Adam + SparseCategoricalCrossentropy + accuracy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def model_summary_string(model: tf.keras.Model) -> str:
    """Devuelve el summary como string (util para guardar al disco)."""
    lines: list[str] = []
    model.summary(print_fn=lambda s: lines.append(s))
    return "\n".join(lines)
