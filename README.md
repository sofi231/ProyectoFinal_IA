# Asistente Robótico por Comandos de Voz — Modalidad C

Proyecto Final · Inteligencia Artificial · Universidad Rafael Landívar · Primer Semestre 2026

Sistema de reconocimiento de comandos de voz en español que controla un **panel de domótica** (modalidad C). Todo el pipeline — captura, preprocesamiento, extracción de features, entrenamiento e inferencia — corre offline, sin APIs externas ni modelos preentrenados de terceros.

---

## Tabla de contenidos

- [Comandos y mapeo a actuadores](#comandos-y-mapeo-a-actuadores)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Instrucciones de grabación](#instrucciones-de-grabación)
- [Ejecución del pipeline](#ejecución-del-pipeline)
- [Archivos generados](#archivos-generados)
- [Decisiones técnicas](#decisiones-técnicas)
- [Estado actual y pendientes](#estado-actual-y-pendientes)

---

## Comandos y mapeo a actuadores

El sistema reconoce **7 clases**:

| Clase | Tipo | Acción sobre el panel |
|---|---|---|
| `ENCIENDE` | acción global | Enciende **todos** los dispositivos |
| `APAGA` | acción global | Apaga **todos** los dispositivos |
| `LUZ` | toggle | Conmuta el LED de iluminación |
| `VENTILADOR` | toggle | Conmuta el ventilador |
| `CERRADURA` | toggle | Conmuta la cerradura simulada |
| `PANEL` | toggle | Conmuta un dispositivo extra del panel |
| `RUIDO` | rechazo | El sistema no ejecuta nada (clase de rechazo) |

---

## Estructura del repositorio

```
ProyectoFinal_IA/
├── Audios/                       # Corpus crudo (.wav) organizado por clase
│   ├── APAGA/
│   ├── CERRADURA/
│   └── ...
├── data/
│   └── processed/                # Datasets listos para entrenar (.npz)
├── models/                       # Modelos entrenados (.h5, .tflite) [pendiente]
├── notebooks/
│   └── 01_exploracion.ipynb      # Inspección de corpus y visualización
├── src/
│   ├── config.py                 # Constantes (SR, MFCC, splits, augmentation)
│   ├── audio_utils.py            # Carga, normalización, VAD por energía
│   ├── features.py               # MFCC + deltas + estandarización
│   ├── augmentation.py           # Time shift, pitch, ruido SNR, SpecAugment
│   └── build_dataset.py          # Pipeline end-to-end
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Instalación

Requiere **Python 3.13** (TensorFlow 2.21 lo soporta; las versiones 3.14+ aún no tienen wheels precompilados).

```bash
cd ProyectoFinal_IA
py -3.13 -m venv .venv

# PowerShell
.\.venv\Scripts\Activate.ps1
# o CMD
.venv\Scripts\activate.bat
# o Git Bash / Linux / macOS
source .venv/Scripts/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencias clave:** TensorFlow 2.21, librosa 0.11, scikit-learn 1.8, soundfile, scipy, matplotlib, tqdm.

---

## Instrucciones de grabación

### Formato del archivo

Cada audio sigue el formato:

```
PALABRA_nombre_numero_entorno.wav
```

| Campo | Valores válidos | Ejemplo |
|---|---|---|
| `PALABRA` | una de las 7 clases (en mayúsculas) | `APAGA`, `LUZ`, `RUIDO` |
| `nombre` | nombre del hablante u alias en minúsculas | `carlos`, `maria` |
| `numero` | índice de toma con cero a la izquierda | `01`, `02`, `15` |
| `entorno` | `silencio`, `ruido`, `ambiente` o `habla` | `silencio` |

Ejemplo válido: `LUZ_dereck_07_ruido.wav`

### Parámetros obligatorios

- **Sample rate:** 16 kHz (mono, PCM 16-bit)
- **Duración:** entre 0.4 s y 1.5 s (el pipeline aplica VAD y zero-padding a 1.5 s)s
- **Ubicación:** `Audios/<CLASE>/<archivo>.wav`

El parser de nombres está en `src/build_dataset.py` (regex `FILENAME_RE`); cualquier archivo que no respete el formato se ignora con un aviso.

### Entornos requeridos

- `silencio`: cuarto cerrado, sin música ni voces de fondo.
- `ruido`: ventilador encendido, voces o tráfico ambiental moderado.
- `ambiente`: solo para la clase `RUIDO` — captura ruido de fondo puro (sin voz).
- `habla`: solo para la clase `RUIDO` — voz pronunciando palabras NO-comando (para que el modelo aprenda a rechazar habla espuria).

---

## Ejecución del pipeline

Desde la raíz del proyecto con el venv activado:

```bash
# Genera el dataset completo con augmentation (3 copias por muestra de train)
python -m src.build_dataset

# Sin augmentation (más rápido, útil para depurar)
python -m src.build_dataset --no-augment

# Cambiar el número de copias aumentadas
python -m src.build_dataset --aug-per-sample 5

# Salida personalizada
python -m src.build_dataset --output data/processed/mi_dataset.npz
```

El script:

1. Recorre `Audios/` recursivamente.
2. Parsea cada nombre de archivo y descarta los inválidos.
3. Hace **split estratificado por clase**: 70% train / 15% val / 15% test (seed fija = 42).
4. Para cada muestra: carga → normaliza pico → **VAD por energía RMS** → zero-pad/recorte centrado a 1.5 s → MFCC (40 coef.) + Δ + ΔΔ = 120 canales × 151 frames → **z-score por clip**.
5. En train, genera N copias aumentadas por muestra (no en val/test).
6. Guarda `.npz` y `.meta.json`.

---

## Archivos generados

### `data/processed/dataset.npz`

| Clave | Forma | Tipo | Descripción |
|---|---|---|---|
| `X_train` | `(N_train, 120, 151)` | float32 | MFCC+Δ+ΔΔ estandarizado |
| `y_train` | `(N_train,)` | int64 | Índice de clase en `classes` |
| `X_val` / `y_val` | igual | igual | Sin augmentation |
| `X_test` / `y_test` | igual | igual | Sin augmentation |
| `classes` | `(7,)` | str | Orden canónico de las clases |
| `sample_rate` | escalar | int32 | 16000 |
| `clip_samples` | escalar | int32 | 24000 (1.5 s × 16 kHz) |

### `data/processed/dataset.meta.json` — el campo `aug`

Archivo paralelo al `.npz` con **un registro por fila** de cada split:

```json
{
  "train": [
    {"file": "APAGA_dereck_01_silencio.wav", "label": "APAGA",
     "speaker": "dereck", "env": "silencio", "aug": "none"},
    {"file": "APAGA_dereck_01_silencio.wav", "label": "APAGA",
     "speaker": "dereck", "env": "silencio", "aug": "v1"},
    {"file": "APAGA_dereck_01_silencio.wav", "label": "APAGA",
     "speaker": "dereck", "env": "silencio", "aug": "v2"},
    ...
  ],
  "val":  [ ... ],
  "test": [ ... ]
}
```

| Valor de `aug` | Significado |
|---|---|
| `"none"` | Muestra **original** procesada (load → normalize → VAD → fix-length → MFCC). |
| `"v1"`, `"v2"`, `"v3"`, … | Copia **aumentada** del archivo. Cada `vN` es una variante distinta generada con una composición aleatoria de time shifting, pitch shifting, inyección de ruido gaussiano y SpecAugment. 

---

## Decisiones técnicas

| Aspecto | Decisión | Justificación |
|---|---|---|
| Features | **MFCC** (40 coef.) + Δ + ΔΔ | Más compacto que mel-spec, decorrelacionado por DCT, deltas aportan dinámica temporal sin disparar dimensión |
| Sample rate | 16 kHz | Estándar para keyword spotting |
| Ventana / hop | 25 ms / 10 ms | Cuasi-estacionariedad de fonemas |
| Clip fijo | 1.5 s (24 000 samples) | Palabras españolas cortas (0.4–1.1 s en el corpus); 1.5 s da margen tras VAD |
| VAD | Energía RMS por frame con umbral por percentil | Simple, explicable matemáticamente, sin dependencias adicionales |
| Normalización | Peak + z-score por clip | Invariante a ganancia del micrófono |
| Split | Estratificado por clase, seed 42 | Mantiene proporciones; reproducible |
| Augmentation | 4 técnicas, solo en train | Time shift, pitch ±2 semitonos, ruido SNR 5–20 dB, SpecAugment |

---

## Modelo base — CNN sobre MFCC

### Arquitectura (`src/model.py`)

CNN 2D que trata la matriz MFCC como imagen tiempo × frecuencia.

```
mfcc_input (120, 151)
  → Reshape (120, 151, 1)
  → Conv2D(32) + BN + ReLU + MaxPool(2,2) + Dropout(0.15)
  → Conv2D(64) + BN + ReLU + MaxPool(2,2) + Dropout(0.25)
  → Conv2D(64) + BN + ReLU + MaxPool(2,2) + Dropout(0.35)
  → GlobalAveragePooling2D
  → Dense(64) + BN + ReLU + Dropout(0.45)
  → Dense(7, softmax)
```

**Total: 61 255 parámetros** (≈ 240 KB en float32).

| Decisión | Justificación |
|---|---|
| Conv2D vs Conv1D | Aprovecha estructura 2D del MFCC; convoluciones espaciales detectan patrones tiempo-frecuencia conjuntos |
| BatchNorm en todas las capas | Estabiliza entrenamiento con dataset pequeño; aporta cierta regularización |
| GlobalAveragePooling2D | Reduce drásticamente parámetros vs Flatten → menos overfitting |
| L2 = 1e-4 + dropout escalonado | Sin regularización el modelo memorizaba (gap train-val de 50 pts); con esto el gap baja a 4 pts |
| LR inicial 5e-4 + Reduce on plateau | LR=1e-3 producía oscilación violenta en val; bajar a 5e-4 estabiliza |
| Class weights balanceados | RUIDO solo tiene 40 muestras vs 144 de los demás; sin esto, el modelo ignora RUIDO |

### Entrenamiento (`src/train.py`)

- **Optimizador**: Adam, lr 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.125e-5 (ReduceLROnPlateau, paciencia 7).
- **EarlyStopping** sobre val_loss, paciencia 20, restaura el mejor checkpoint.
- **Loss**: SparseCategoricalCrossentropy.
- **Batch**: 32. **Epochs**: hasta 120 (con corte temprano).
- **Class weights**: balanceados (sklearn estilo `balanced`).

### Resultados (corpus de 902 muestras, 7 hablantes)

| Split | n | Accuracy | F1 macro |
|---|---|---|---|
| Train | 2520 (con aug) | **0.929** | 0.930 |
| Val | 136 | **0.875** | 0.879 |
| **Test** | **136** | **0.927** | **0.924** |

**Per-class TEST**:

| Clase | Precision | Recall | F1 | n |
|---|---|---|---|---|
| APAGA | 0.952 | 0.952 | 0.952 | 21 |
| CERRADURA | 1.000 | 0.955 | 0.977 | 22 |
| ENCIENDE | 0.808 | 1.000 | 0.894 | 21 |
| LUZ | 0.917 | 1.000 | 0.957 | 22 |
| PANEL | 1.000 | 0.773 | 0.872 | 22 |
| RUIDO | 1.000 | 0.833 | 0.909 | 6 |
| VENTILADOR | 0.909 | 0.909 | 0.909 | 22 |

**Comparativa con baseline anterior** (137 muestras, 1 hablante):

| | Baseline | Final |
|---|---|---|
| Muestras crudas | 137 | 902 |
| Hablantes | 1 (`dereck`) | 7 (incluye 5 voluntarios externos) |
| Test accuracy | 0.519 | **0.927** |
| F1 macro test | 0.476 | **0.924** |
| RUIDO F1 | 0.000 | 0.909 |
| VENTILADOR F1 | 0.000 | 0.909 |

**Errores observados** (10/136 en test):
- `PANEL → ENCIENDE/LUZ` (5 casos): el modelo confunde palabras con cadencia similar.
- `VENTILADOR → ENCIENDE` (2 casos): ambas son palabras largas que terminan en vocal.
- `APAGA → ENCIENDE` (1) y `CERRADURA → VENTILADOR` (1).
- `RUIDO → APAGA` (1): muestra de ruido con ráfaga energética.

### Exports

| Archivo | Tamaño | Uso |
|---|---|---|
| `models/cnn_voz_base.keras` | 807 KB | Inferencia con TensorFlow / Keras |
| `models/cnn_voz_base.tflite` | 242 KB | Inferencia float32 con TFLite (PC o Raspberry Pi) |
| `models/cnn_voz_base_int8.tflite` | 72 KB | Cuantizado INT8 para microcontroladores (ESP32, TFLite Micro) |

Todos los artefactos auxiliares — `training_history.json`, `training_report.json`, `confusion_matrix.npy`, `training_curves.png`, `confusion_matrix.png`, `model_summary.txt` — se generan junto con el modelo en cada corrida.

### Cómo reproducir

```bash
# 1. Pipeline de features
python -m src.build_dataset

# 2. Entrenamiento (≈ 70 minutos en CPU con esta config)
python -m src.train

# 3. Visualización de curvas y matriz (genera PNG en models/)
python -m src.train --help     # opciones disponibles
```

---

## Inferencia en tiempo real

Pipeline streaming desde micrófono: captura → ring buffer → VAD por energía → segmentación → MFCC → inferencia → decisión (`src/realtime.py`).

### Uso

```bash
# Listar dispositivos de entrada
python -m src.realtime --list-devices

# Inferencia con TFLite (recomendado, más rápido)
python -m src.realtime

# Forzar Keras (.keras)
python -m src.realtime --model models/cnn_voz_base.keras

# Cuantizado INT8 para perfiles tipo microcontrolador
python -m src.realtime --model models/cnn_voz_base_int8.tflite

# Elegir micrófono específico y umbral VAD manual
python -m src.realtime --device 1 --threshold 0.02
```

### Cómo funciona el VAD streaming

1. **Calibración**: al arrancar, captura 1 s de silencio y mide RMS de fondo. Umbral = RMS × 4 (configurable con `--threshold-factor`).
2. **Estado IDLE**: mantiene un *pre-roll* de 200 ms en un buffer circular para no perder el inicio del fonema.
3. **Transición a ACTIVO**: cuando ≥ N frames consecutivos superan el umbral (mínimo 200 ms de habla).
4. **Estado ACTIVO**: acumula samples del segmento.
5. **Transición a IDLE**: cuando ≥ 400 ms de silencio o el segmento llega a 2 s (corte de seguridad).
6. **Inferencia**: segmento → peak-normalize → padding/recorte a 1.5 s → MFCC → TFLite → softmax.
7. **Decisión**:
   - Si `confidence < 0.50` → ignorado (umbral configurable con `--confidence-min`).
   - Si `label == "RUIDO"` → ignorado (rechazo explícito).
   - En cualquier otro caso → ejecutar acción (printa `[ACCION: LUZ]`, etc.).

### Latencia medida (CPU, sin micrófono)

Benchmark con 30 inferencias sobre clips sintéticos de 1.5 s:

| Backend | Features (p95) | Inferencia (p95) | **Total (p95)** | Margen vs 500 ms |
|---|---|---|---|---|
| Keras (`.keras`) | 21 ms | 301 ms | 323 ms | 177 ms |
| TFLite float32 | 20 ms | 15 ms | **32 ms** | **468 ms** |
| TFLite INT8 | 21 ms | 7 ms | **25 ms** | **475 ms** |

Sobre 20 muestras reales del test: **20/20 aciertos** en los tres backends. El TFLite produce idénticas predicciones que Keras pero ≈10× más rápido.

> El PDF exige < 500 ms end-to-end. Con TFLite + VAD el sistema queda con > 460 ms de margen, suficiente para sumar captura del micrófono y el accionamiento de hardware GPIO/serial.

### Salida típica

```
Cargando modelo cnn_voz_base.tflite ...
  backend: TFLite
[calibracion] mantenga silencio durante 1.0 s ...
[calibracion] ruido RMS=0.00321  umbral=0.01284

Escuchando en device=None  sr=16000 Hz  umbral=0.01284
Habla un comando. Ctrl+C para salir.

  -> ENCIENDE     conf=0.97 ******************    dur=820ms  feat=18.2ms  inf=12.4ms  total=30.6ms  [ACCION: ENCIENDE]
  -> LUZ          conf=0.94 ******************    dur=540ms  feat=17.1ms  inf=11.8ms  total=28.9ms  [ACCION: LUZ]
  -> RUIDO        conf=0.81 ****************      dur=910ms  feat=18.4ms  inf=12.0ms  total=30.4ms  [ignorado: rechazo]
```

### Hooks de hardware

`RealtimeRecognizer._decide_action()` es el punto de extensión. Para conectar al panel físico (modalidad C):

```python
def _decide_action(self, label, conf):
    if conf < self.confidence_min or label == "RUIDO":
        return "[ignorado]"
    # Ejemplo: enviar comando por serial al ESP32 del panel
    import serial
    self.ser.write(f"{label}\n".encode())
    return f"[ACCION: {label}]"
```

---
