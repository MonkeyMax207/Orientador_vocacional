/*
 * Oso Andino — control de ojos y boca para el orientador vocacional de la UAO.
 *
 * Basado en el test de animaciones de Juan Diego, con los MISMOS limites calibrados: esos
 * numeros salieron de probar el mecanismo y no se tocan desde el lado del computador. La Pi
 * habla en porcentajes (0-100) y aqui se traducen a microsegundos.
 *
 * Que cambia respecto al test, y por que:
 *
 *   1. NADA BLOQUEA. El test usaba delay() dentro de las animaciones. Con la boca siguiendo la
 *      voz unas 21 veces por segundo, cualquier delay() se traga esas actualizaciones y la
 *      mandibula se queda abierta a destiempo. Todo es una maquina de estados con millis().
 *
 *   2. PROTOCOLO DE LINEAS en vez de un caracter de menu. La Pi necesita decir "boca al 37%",
 *      no "ejecuta la animacion 6".
 *
 *   3. VIDA PROPIA. El oso parpadea y mueve la mirada solo, sin que nadie se lo pida. Importa
 *      mas de lo que parece: en la Pi un turno tarda entre 5 y 15 segundos, y un animatronico
 *      inmovil todo ese rato parece apagado. La Pi puede estar pensando; el oso no.
 *
 *   4. HOMBRE MUERTO. Si deja de llegar M (boca) por medio segundo, la boca se cierra sola.
 *      Si el cable se suelta o el programa muere a mitad de frase, el oso no se queda con la
 *      mandibula colgando.
 *
 * Protocolo (una orden por linea, terminada en \n):
 *   M<0-100>        boca: 0 cerrada, 100 abierta
 *   E<0-100>        parpados: 0 abiertos, 100 cerrados
 *   L<pan>,<tilt>   mirada, 0-100 cada eje
 *   B               un parpadeo
 *   R               reposo: ojos abiertos, mirada al centro, boca cerrada
 *   A<nombre>       animacion con nombre: saludar, asentir, inclinar_cabeza, despedir, pensar
 *   I<0|1>          vida propia (parpadeo y mirada automaticos) apagada o encendida
 *   P               ping; responde "OK" (la Pi lo usa para saber que hay alguien)
 */

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define SDA_PIN 21
#define SCL_PIN 22
#define SERVO_FREQ 50

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

struct ServoConfig {
  bool activo;
  int us_min;   // 0%
  int us_max;   // 100%
};

ServoConfig limites[16];

// Canales, iguales que en el test original.
const int CH_PARPADO_SUP_IZQ = 0;
const int CH_PARPADO_SUP_DER = 1;
const int CH_PARPADO_INF_IZQ = 2;
const int CH_PARPADO_INF_DER = 3;
const int CH_PAN  = 4;
const int CH_TILT = 5;
const int CH_BOCA = 6;

// --- Estado ---
int  bocaActual   = 0;
unsigned long ultimaBoca = 0;       // para el hombre muerto
const unsigned long BOCA_TIMEOUT_MS = 500;

bool vidaPropia = true;
unsigned long proximoParpadeo = 0;
unsigned long proximaMirada   = 0;

// Parpadeo no bloqueante: 0 = quieto, 1 = cerrando, 2 = esperando, 3 = abriendo
int  faseParpadeo = 0;
unsigned long tParpadeo = 0;
const unsigned long PARPADEO_CIERRE_MS = 90;

// Animacion con nombre, tambien por pasos.
int  animPaso = 0;
unsigned long tAnim = 0;
String animActual = "";

String entrada = "";

void inicializarLimites() {
  for (int i = 0; i < 16; i++) limites[i].activo = false;

  // -- OJOS: 0% = Abierto | 100% = Cerrado --
  limites[0] = {true, 925, 1900};   // CH0: Parpado sup IZQ
  limites[1] = {true, 1850, 700};   // CH1: Parpado sup DER (INVERTIDO)
  limites[2] = {true, 2250, 1000};  // CH2: Parpado inf IZQ (INVERTIDO)
  limites[3] = {true, 1050, 1950};  // CH3: Parpado inf DER
  limites[4] = {true, 825, 1375};   // CH4: Ojos PAN
  limites[5] = {true, 500, 1525};   // CH5: Ojos TILT

  // -- BOCA: 0% = Cerrada | 100% = Abierta --
  limites[6] = {true, 1300, 1700};  // CH6: Boca (Mini servo)
}

void mover(int canal, int porcentaje) {
  if (!limites[canal].activo) return;
  porcentaje = constrain(porcentaje, 0, 100);
  int us = map(porcentaje, 0, 100, limites[canal].us_min, limites[canal].us_max);
  pwm.writeMicroseconds(canal, us);
}

void ponerParpados(int cierre) {
  mover(CH_PARPADO_SUP_IZQ, cierre);
  mover(CH_PARPADO_SUP_DER, cierre);
  mover(CH_PARPADO_INF_IZQ, cierre);
  mover(CH_PARPADO_INF_DER, cierre);
}

void ponerBoca(int nivel) {
  bocaActual = constrain(nivel, 0, 100);
  mover(CH_BOCA, bocaActual);
  ultimaBoca = millis();
}

void reposo() {
  ponerParpados(0);
  mover(CH_PAN, 50);
  mover(CH_TILT, 50);
  ponerBoca(0);
  faseParpadeo = 0;
  animActual = "";
}

void empezarParpadeo() {
  if (faseParpadeo != 0) return;        // ya hay uno en curso
  faseParpadeo = 1;
  tParpadeo = millis();
}

// Avanza el parpadeo sin bloquear a nadie.
void pasoParpadeo() {
  if (faseParpadeo == 0) return;
  unsigned long ahora = millis();
  if (faseParpadeo == 1) {
    ponerParpados(100);
    faseParpadeo = 2;
    tParpadeo = ahora;
  } else if (faseParpadeo == 2 && ahora - tParpadeo >= PARPADEO_CIERRE_MS) {
    ponerParpados(0);
    faseParpadeo = 0;
  }
}

void empezarAnimacion(const String &nombre) {
  animActual = nombre;
  animPaso = 0;
  tAnim = millis();
}

// Las animaciones con nombre son las que ya conoce animations.py del lado de la Pi.
void pasoAnimacion() {
  if (animActual == "") return;
  unsigned long dt = millis() - tAnim;

  if (animActual == "saludar") {                 // mirada de lado a lado, alegre
    if (animPaso == 0 && dt > 0)   { mover(CH_PAN, 15); animPaso = 1; tAnim = millis(); }
    else if (animPaso == 1 && dt > 350) { mover(CH_PAN, 85); animPaso = 2; tAnim = millis(); }
    else if (animPaso == 2 && dt > 350) { mover(CH_PAN, 50); animActual = ""; }
  } else if (animActual == "asentir") {          // dos cabeceos con el tilt
    if (animPaso == 0 && dt > 0)   { mover(CH_TILT, 25); animPaso = 1; tAnim = millis(); }
    else if (animPaso == 1 && dt > 220) { mover(CH_TILT, 70); animPaso = 2; tAnim = millis(); }
    else if (animPaso == 2 && dt > 220) { mover(CH_TILT, 50); animActual = ""; }
  } else if (animActual == "inclinar_cabeza") {  // curiosidad: mirada arriba y a un lado
    if (animPaso == 0 && dt > 0)   { mover(CH_TILT, 30); mover(CH_PAN, 35); animPaso = 1; tAnim = millis(); }
    else if (animPaso == 1 && dt > 700) { mover(CH_TILT, 50); mover(CH_PAN, 50); animActual = ""; }
  } else if (animActual == "despedir") {
    if (animPaso == 0 && dt > 0)   { empezarParpadeo(); mover(CH_PAN, 20); animPaso = 1; tAnim = millis(); }
    else if (animPaso == 1 && dt > 450) { mover(CH_PAN, 50); animActual = ""; }
  } else if (animActual == "pensar") {           // mirada arriba, como buscando la respuesta
    if (animPaso == 0 && dt > 0)   { mover(CH_TILT, 20); mover(CH_PAN, 70); animPaso = 1; tAnim = millis(); }
    else if (animPaso == 1 && dt > 900) { mover(CH_PAN, 35); animPaso = 2; tAnim = millis(); }
    else if (animPaso == 2 && dt > 900) { mover(CH_TILT, 50); mover(CH_PAN, 50); animActual = ""; }
  } else {
    animActual = "";                             // nombre desconocido: no hacer nada raro
  }
}

// Parpadeo y mirada espontaneos: el oso parece vivo mientras la Pi piensa.
void pasoVidaPropia() {
  if (!vidaPropia) return;
  unsigned long ahora = millis();
  if (ahora > proximoParpadeo) {
    empezarParpadeo();
    proximoParpadeo = ahora + random(2600, 6200);    // como una persona, irregular
  }
  if (ahora > proximaMirada && animActual == "") {
    mover(CH_PAN,  random(35, 66));                  // desvios pequenos: mirar, no girar
    mover(CH_TILT, random(40, 61));
    proximaMirada = ahora + random(3500, 9000);
  }
}

// Si la Pi deja de mandar boca, cerrarla: nadie se queda con la mandibula colgando.
void pasoHombreMuerto() {
  if (bocaActual > 0 && millis() - ultimaBoca > BOCA_TIMEOUT_MS) {
    ponerBoca(0);
  }
}

void procesar(const String &linea) {
  if (linea.length() == 0) return;
  char tipo = linea.charAt(0);
  String resto = linea.substring(1);
  resto.trim();

  switch (tipo) {
    case 'M': ponerBoca(resto.toInt()); break;
    case 'E': ponerParpados(constrain(resto.toInt(), 0, 100)); break;
    case 'B': empezarParpadeo(); break;
    case 'R': reposo(); break;
    case 'A': empezarAnimacion(resto); break;
    case 'I': vidaPropia = (resto.toInt() != 0); break;
    case 'P': Serial.println("OK"); break;
    case 'L': {
      int coma = resto.indexOf(',');
      if (coma > 0) {
        mover(CH_PAN,  constrain(resto.substring(0, coma).toInt(), 0, 100));
        mover(CH_TILT, constrain(resto.substring(coma + 1).toInt(), 0, 100));
      }
      break;
    }
    default: break;        // callado a proposito: la Pi no debe recibir ruido que no pidio
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  inicializarLimites();
  randomSeed(esp_random());
  reposo();

  proximoParpadeo = millis() + 2000;
  proximaMirada   = millis() + 3000;
  Serial.println("OSO LISTO");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      procesar(entrada);
      entrada = "";
    } else if (entrada.length() < 48) {     // una orden larga es ruido: no crecer sin limite
      entrada += c;
    }
  }
  pasoParpadeo();
  pasoAnimacion();
  pasoVidaPropia();
  pasoHombreMuerto();
}
