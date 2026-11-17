"""El puente con el ESP32 que mueve los ojos y la boca del oso.

Hardware (segun el firmware de Juan Diego): ESP32 + PCA9685 por I2C en 0x40, siete canales.
Ojos: CH0-CH3 parpados, CH4 pan, CH5 tilt. Boca: CH6, un mini servo. Los limites en
microsegundos estan calibrados en el firmware y no se tocan desde aqui: esta clase habla en
porcentajes (0-100) y el ESP32 los traduce. Si el mecanismo cambia, se cambia en un solo sitio.

Tres decisiones que valen la pena explicar:

**Ausente no es un error.** Si no hay ESP32 conectado, todo esto se vuelve silencioso y el
robot sigue hablando igual. Hoy mismo (2026-11-17) no hay ninguno enchufado a la Pi, y nadie
deberia quedarse sin voz porque falte un servo.

**Las escrituras no bloquean al audio.** La boca se actualiza ~21 veces por segundo desde el
hilo que reproduce el sonido; si una escritura al puerto serie se demorara, la voz se cortaria.
Por eso hay un hilo escritor aparte y el valor de boca se SOBREESCRIBE en vez de encolarse:
llegar tarde con la posicion de una mandibula no sirve de nada, solo importa la ultima.

**Lo repetido no se envia.** Un servo al que se le manda la misma posicion veinte veces por
segundo zumba y se calienta. Solo se escribe cuando el valor cambia de verdad.
"""
import threading
import time

# Canales, tal como los define el firmware.
PARPADOS = (0, 1, 2, 3)
PAN, TILT, BOCA = 4, 5, 6

# Cambios de boca menores a esto no se envian: el servo no los distingue y el puerto si los paga.
UMBRAL_BOCA = 4


class Animatronico:
    """Habla con el ESP32 por el puerto serie. Sin puerto, no hace nada y no se queja dos veces."""

    def __init__(self, puerto: str = "", baudios: int = 115200, abrir=None, log=print):
        self.log = log
        self.serie = None
        self._ultima_boca = -1
        self._pendiente = None          # ultimo valor de boca por enviar (se sobreescribe)
        self._cola: list[str] = []      # ordenes sueltas (parpadeo, mirada): estas no se pisan
        self._cv = threading.Condition()
        self._vivo = True
        if not puerto:
            self.log("[esp32] sin puerto configurado: el robot habla sin moverse")
            return
        try:
            self.serie = (abrir or self._abrir_serie)(puerto, baudios)
        except Exception as e:
            # Un servo que falta no puede dejar mudo al robot.
            self.log(f"[esp32] no se pudo abrir {puerto} ({e}); se sigue sin movimiento")
            self.serie = None
            return
        threading.Thread(target=self._escritor, daemon=True).start()
        self.log(f"[esp32] conectado en {puerto}")

    @staticmethod
    def _abrir_serie(puerto: str, baudios: int):
        import serial                                  # solo si de verdad hay hardware
        s = serial.Serial(puerto, baudios, timeout=0.2, write_timeout=0.2)
        time.sleep(2.0)                                # el ESP32 se reinicia al abrir el puerto
        s.reset_input_buffer()
        return s

    @property
    def conectado(self) -> bool:
        return self.serie is not None

    # ---------------------------------------------------------------- ordenes
    def boca(self, nivel: int) -> None:
        """0 = cerrada, 100 = abierta del todo. Se llama ~21 veces por segundo al hablar."""
        nivel = max(0, min(100, int(nivel)))
        if nivel == self._ultima_boca:
            return                                     # ya esta ahi: no hay nada que decir
        if abs(nivel - self._ultima_boca) < UMBRAL_BOCA and nivel not in (0, 100):
            return                                     # el servo no distingue ese cambio

        self._ultima_boca = nivel
        self._enviar(f"M{nivel}", pisable=True)

    def animacion(self, nombre: str) -> None:
        """Las mismas que ya conoce animations.py: saludar, asentir, inclinar_cabeza,
        despedir, pensar. El ESP32 las ejecuta por pasos, sin bloquear la boca."""
        if nombre:
            self._enviar(f"A{nombre}")

    def parpadear(self) -> None:
        self._enviar("B")

    def mirar(self, pan: int, tilt: int) -> None:
        self._enviar(f"L{max(0, min(100, int(pan)))},{max(0, min(100, int(tilt)))}")

    def parpados(self, cierre: int) -> None:
        """0 = ojos abiertos, 100 = cerrados."""
        self._enviar(f"E{max(0, min(100, int(cierre)))}")

    def reposo(self) -> None:
        """Devuelve al oso a su postura neutra: ojos abiertos, mirada al centro, boca cerrada."""
        self._enviar("R")
        self._ultima_boca = 0

    def cerrar(self) -> None:
        self._vivo = False
        with self._cv:
            self._cv.notify_all()
        if self.serie:
            try:
                self.serie.close()
            except Exception:
                pass

    # ---------------------------------------------------------------- interno
    def _enviar(self, orden: str, pisable: bool = False) -> None:
        if not self.serie:
            return
        with self._cv:
            if pisable:
                self._pendiente = orden     # solo importa la ultima posicion de la boca
            else:
                self._cola.append(orden)
            self._cv.notify()

    def _escritor(self) -> None:
        while self._vivo:
            with self._cv:
                while self._vivo and not self._cola and self._pendiente is None:
                    self._cv.wait(0.5)
                if not self._vivo:
                    return
                ordenes = self._cola[:]
                self._cola.clear()
                if self._pendiente is not None:
                    ordenes.append(self._pendiente)
                    self._pendiente = None
            for orden in ordenes:
                try:
                    self.serie.write((orden + "\n").encode("ascii"))
                except Exception as e:
                    self.log(f"[esp32] se perdio la conexion ({e}); se sigue sin movimiento")
                    self.serie = None
                    return


def nivel_de_boca(muestras, pico_ref: int = 9000) -> int:
    """Cuanto abrir la boca para este trozo de audio, de 0 a 100.

    Se usa la amplitud media absoluta y no el RMS: es mas barata de calcular (esto corre en el
    hilo del audio, 21 veces por segundo) y para una mandibula la diferencia no se ve.

    `pico_ref` es la amplitud que corresponde a la boca abierta del todo. Piper entrega int16
    con picos tipicos de 8.000-12.000 en voz normal, asi que 9.000 deja la boca usando casi
    todo su recorrido sin quedarse pegada arriba.
    """
    if muestras is None or len(muestras) == 0:
        return 0
    try:
        import numpy as np
        media = float(np.abs(np.asarray(muestras, dtype="float32")).mean())
    except Exception:                      # sin numpy, a mano: esto nunca debe tumbar la voz
        media = sum(abs(int(v)) for v in muestras) / len(muestras)
    return max(0, min(100, int(media * 100 / pico_ref)))


def make_animatronico(cfg, log=print) -> "Animatronico":
    """Crea el puente segun la configuracion del perfil. Sin `esp32_port`, queda inerte."""
    return Animatronico(getattr(cfg, "esp32_port", ""),
                        getattr(cfg, "esp32_baud", 115200), log=log)
