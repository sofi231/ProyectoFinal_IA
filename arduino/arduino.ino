// =============================================================
// Proyecto Final - Inteligencia Artificial
// Universidad Rafael Landivar - Primer Semestre 2026
// Panel de Domotica Controlado por Voz
// VERSION CON BUZZER (en lugar de servo)
// =============================================================

// -------- Pines de salida --------
const int PIN_RELE_1 = 7;  // Luz principal (LED 1)
const int PIN_RELE_2 = 8;  // Ventilador
const int PIN_RELE_3 = 12; // Panel (LED 2)
const int PIN_RELE_4 = 13; // Reserva
const int PIN_BUZZER = 10; // Buzzer (cerradura simulada)

// -------- Logica del modulo de reles --------
const int RELE_ON = LOW;
const int RELE_OFF = HIGH;

// -------- Objetos globales --------
String comando = "";

// -------- Estado actual del sistema --------
bool estadoLuz = false;
bool estadoVentilador = false;
bool estadoPanel = false;
bool estadoCerradura = false;

// =============================================================
// FUNCIONES AUXILIARES
// =============================================================

void prenderLuz() {
  digitalWrite(PIN_RELE_1, RELE_ON);
  estadoLuz = true;
}
void apagarLuz() {
  digitalWrite(PIN_RELE_1, RELE_OFF);
  estadoLuz = false;
}
void prenderVentilador() {
  digitalWrite(PIN_RELE_2, RELE_ON);
  estadoVentilador = true;
}
void apagarVentilador() {
  digitalWrite(PIN_RELE_2, RELE_OFF);
  estadoVentilador = false;
}
void prenderPanel() {
  digitalWrite(PIN_RELE_3, RELE_ON);
  estadoPanel = true;
}
void apagarPanel() {
  digitalWrite(PIN_RELE_3, RELE_OFF);
  estadoPanel = false;
}

// Sonido de "cerradura abriendo": 2 beeps cortos
void sonarAbrir() {
  digitalWrite(PIN_BUZZER, HIGH);
  delay(100);
  digitalWrite(PIN_BUZZER, LOW);
  delay(80);
  digitalWrite(PIN_BUZZER, HIGH);
  delay(100);
  digitalWrite(PIN_BUZZER, LOW);
  estadoCerradura = true;
}

// Sonido de "cerradura cerrando": 1 beep largo
void sonarCerrar() {
  digitalWrite(PIN_BUZZER, HIGH);
  delay(300);
  digitalWrite(PIN_BUZZER, LOW);
  estadoCerradura = false;
}

// =============================================================
// SETUP
// =============================================================
void setup() {
  Serial.begin(9600);

  pinMode(PIN_RELE_1, OUTPUT);
  pinMode(PIN_RELE_2, OUTPUT);
  pinMode(PIN_RELE_3, OUTPUT);
  pinMode(PIN_RELE_4, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);

  digitalWrite(PIN_RELE_1, RELE_OFF);
  digitalWrite(PIN_RELE_2, RELE_OFF);
  digitalWrite(PIN_RELE_3, RELE_OFF);
  digitalWrite(PIN_RELE_4, RELE_OFF);
  digitalWrite(PIN_BUZZER, LOW);

  comando.reserve(32);

  Serial.println("=============================");
  Serial.println("Sistema Iniciado");
  Serial.println("Panel de Domotica - URL 2026");
  Serial.println("Comandos disponibles:");
  Serial.println("  LUZ, PANEL, VENTILADOR, CERRADURA (toggle)");
  Serial.println("  ENCIENDE (prende todo), APAGA (apaga todo)");
  Serial.println("  STATUS, PING");
  Serial.println("=============================");
}

// =============================================================
// LOOP
// =============================================================
void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (comando.length() > 0) {
        comando.trim();
        comando.toUpperCase();
        procesarComando(comando);
        comando = "";
      }
    } else {
      comando += c;
    }
  }
}

// =============================================================
// PROCESAR COMANDO
// =============================================================
void procesarComando(String cmd) {
  if (cmd == "LUZ") {
    if (estadoLuz) {
      apagarLuz();
      Serial.println("OK: Luz apagada");
    } else {
      prenderLuz();
      Serial.println("OK: Luz encendida");
    }
  } else if (cmd == "PANEL") {
    if (estadoPanel) {
      apagarPanel();
      Serial.println("OK: Panel apagado");
    } else {
      prenderPanel();
      Serial.println("OK: Panel encendido");
    }
  } else if (cmd == "VENTILADOR") {
    if (estadoVentilador) {
      apagarVentilador();
      Serial.println("OK: Ventilador apagado");
    } else {
      prenderVentilador();
      Serial.println("OK: Ventilador encendido");
    }
  } else if (cmd == "CERRADURA") {
    if (estadoCerradura) {
      sonarCerrar();
      Serial.println("OK: Cerradura cerrada");
    } else {
      sonarAbrir();
      Serial.println("OK: Cerradura abierta");
    }
  } else if (cmd == "ENCIENDE") {
    prenderLuz();
    prenderVentilador();
    prenderPanel();
    sonarAbrir();
    Serial.println("OK: Todo encendido");
  } else if (cmd == "APAGA") {
    apagarLuz();
    apagarVentilador();
    apagarPanel();
    sonarCerrar();
    Serial.println("OK: Todo apagado");
  } else if (cmd == "PING") {
    Serial.println("PONG");
  } else if (cmd == "STATUS") {
    Serial.print("Luz: ");
    Serial.println(estadoLuz ? "ON" : "OFF");
    Serial.print("Ventilador: ");
    Serial.println(estadoVentilador ? "ON" : "OFF");
    Serial.print("Panel: ");
    Serial.println(estadoPanel ? "ON" : "OFF");
    Serial.print("Cerradura: ");
    Serial.println(estadoCerradura ? "ABIERTA (sono)" : "CERRADA (sono)");
  } else {
    Serial.print("ERROR: Comando desconocido: ");
    Serial.println(cmd);
  }
}