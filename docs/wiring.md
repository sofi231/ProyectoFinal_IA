# Conexiones del Panel de Domótica

Proyecto Final de Inteligencia Artificial - Universidad Rafael Landívar - Primer Semestre 2026.

Modalidad C: Panel de domótica controlado por voz. Configuración B: Arduino UNO + laptop.

## Arduino UNO

| Pin Arduino | Va a | Función |
|---|---|---|
| Pin digital 7 | IN1 del módulo de relés | Señal de control relé 1 (luz principal) |
| Pin digital 8 | IN2 del módulo de relés | Señal de control relé 2 (ventilador) |
| Pin digital 9 | Pata + del buzzer | Señal de activación del buzzer (cerradura) |
| Pin digital 12 | IN3 del módulo de relés | Señal de control relé 3 (panel) |
| Pin digital 13 | IN4 del módulo de relés | Señal de control relé 4 (reserva) |
| Pin 5V | VCC del módulo de relés | Alimentación lógica del módulo |
| Pin GND | GND del módulo de relés | Tierra del módulo |
| Pin GND | Pata − del buzzer | Tierra del buzzer (directo, no por protoboard) |
| Pin GND | Riel − de la protoboard | Puente de tierra común con el resto del sistema |
| Puerto USB | Laptop | Comunicación serial y alimentación del Arduino |

## Fuente externa 12V

| Origen | Destino |
|---|---|
| Plug barrel macho de la fuente | Adaptador plug a bornera |
| Tornillo + del adaptador | IN+ del LM2596 y Riel + inferior de la protoboard (dos cables al mismo tornillo) |
| Tornillo − del adaptador | IN− del LM2596 y Riel − de la protoboard (dos cables al mismo tornillo) |

## LM2596 con display

| Pin LM2596 | Va a | Voltaje |
|---|---|---|
| IN+ | Tornillo + del adaptador plug a bornera | 12V de entrada |
| IN− | Tornillo − del adaptador plug a bornera | GND |
| OUT+ | Riel + superior de la protoboard | 5V calibrado |
| OUT− | Riel − de la protoboard | GND |

Calibrado a 5.1V mediante el tornillo dorado de ajuste.

## Protoboard

| Riel | Voltaje | Recibe de | Distribuye a |
|---|---|---|---|
| Riel + SUPERIOR | 5V | OUT+ del LM2596 | COM del relé 1, COM del relé 3 |
| Riel + INFERIOR | 12V | + del adaptador plug a bornera | COM del relé 2 |
| Riel − (superior e inferior unidos) | GND | OUT− del LM2596, − del adaptador, GND del Arduino | Cátodos de LEDs, − del ventilador |

Los dos rieles − están unidos entre sí con un cable puente.

## Relé 1 - LED 1 (luz principal)

| Terminal del relé 1 | Va a |
|---|---|
| NC (tornillo izquierdo) | Sin conectar |
| COM (tornillo del medio) | Riel + superior (5V) |
| NO (tornillo derecho) | Una pata de la resistencia 220Ω |
| Otra pata de la resistencia | Ánodo (pata larga) del LED 1 |
| Cátodo (pata corta) del LED 1 | Riel − |

## Relé 2 - Ventilador 12V

| Terminal del relé 2 | Va a |
|---|---|
| NC (tornillo izquierdo) | Sin conectar |
| COM (tornillo del medio) | Riel + inferior (12V) |
| NO (tornillo derecho) | Cable rojo del ventilador |
| Cable negro del ventilador | Riel − |
| Tercer cable del ventilador (si existe) | Aislado con cinta, sin conectar |

## Relé 3 - LED 2 (panel)

| Terminal del relé 3 | Va a |
|---|---|
| NC (tornillo izquierdo) | Sin conectar |
| COM (tornillo del medio) | Riel + superior (5V) |
| NO (tornillo derecho) | Una pata de la segunda resistencia 220Ω |
| Otra pata de la resistencia | Ánodo (pata larga) del LED 2 |
| Cátodo (pata corta) del LED 2 | Riel − |

## Relé 4 - Reserva

Sin conectar en el lado de potencia. Disponible para uso futuro.

## Buzzer (cerradura simulada)

| Pata del buzzer | Va a |
|---|---|
| Pata + | Pin digital 9 del Arduino |
| Pata − | Pin GND del Arduino (directo, no por protoboard) |

Importante: el GND del buzzer va directamente al pin GND del Arduino y no al riel − de la protoboard, debido a inestabilidad detectada en el riel.

## Mapeo de comandos del software

| Comando serial | Pin afectado | Acción física |
|---|---|---|
| LUZ | Pin 7 (relé 1) | Toggle LED 1 |
| PANEL | Pin 12 (relé 3) | Toggle LED 2 |
| VENTILADOR | Pin 8 (relé 2) | Toggle ventilador |
| CERRADURA | Pin 9 (buzzer) | Beep abrir (2 cortos) o cerrar (1 largo) |
| ENCIENDE | Pines 7, 8, 12, 9 | Activa todos los actuadores |
| APAGA | Pines 7, 8, 12, 9 | Desactiva todos los actuadores |
| STATUS | Ninguno | Reporta estado actual por serial |
| PING | Ninguno | Responde "PONG" |

Los comandos individuales (LUZ, PANEL, VENTILADOR, CERRADURA) operan como toggle: alternan el estado del actuador entre encendido y apagado en cada activación.

## Notas del módulo de relés

- Modelo: SRD-05VDC-SL-C, 4 canales.
- Lógica: activo en bajo (HIGH apaga el relé, LOW lo activa).
- Cada relé tiene tres terminales de tornillo en el lado de potencia: NC (izquierdo), COM (medio), NO (derecho).
- El módulo se alimenta con 5V del Arduino para la parte lógica.