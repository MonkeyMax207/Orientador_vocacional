"""Filler phrases: pre-generated clips that play the instant the user stops talking.

They hide the STT + LLM delay (the "Mmm, déjame pensar..." a person would say).
Generate once per voice:  uv run python fillers.py [pc|pi]
Clips go to models/fillers/<voice>/ (ignored by git, rebuilt by this script).
"""
import random
from pathlib import Path

from audio_io import load_wav, save_wav

FILLER_DIR = Path(__file__).parent / "models" / "fillers"

# Suenan apenas la persona deja de hablar (filler_after_s = 0), junto con la animacion de
# pensar. Su trabajo es tapar los ~7,6 s que toma un turno en la Pi: 0,6 de espera, 1,9 de
# STT, 4,5 del modelo y 0,6 de voz. Medido el 2026-11-17 — el modelo es el 57% y no baja mas
# sin perder calidad, asi que lo que queda por mejorar es la espera PERCIBIDA, no la real.
#
# Solo palabras de verdad: espeak deletrea "Mmm" como "eme eme eme". Nada de digitos.
# Cortos a proposito (1-2 s): reconocen que te oyo y dejan sitio a la respuesta.
FILLERS = [
    "Uy, qué pregunta tan interesante.",
    "A ver, déjame pensar.",
    "Claro, déjame revisar.",
    "Buena pregunta, a ver.",
    "Ah, eso sí está bueno.",
    "Dame un segundito.",
]
# Solo si a los slow_llm_s todavia no hay respuesta. Mas largos: aqui la persona ya sabe que
# se esta demorando, y lo que ayuda es que el robot lo reconozca en vez de callarse.
SLOW_FILLERS = [
    "Déjame ver qué encuentro en mis pensamientos.",
    "Un momentito, que quiero darte el dato correcto.",
    "Ya casi, déjame terminar de pensarlo.",
    "Espérame tantico, estoy buscando bien.",
]


class Fillers:
    def __init__(self, voice: str):
        d = FILLER_DIR / voice
        self.normal = [load_wav(p)[0] for p in sorted(d.glob("filler_*.wav"))]
        self.slow = [load_wav(p)[0] for p in sorted(d.glob("slow_*.wav"))]
        if not self.normal or not self.slow:
            raise SystemExit(
                f"Faltan los fillers de '{voice}'. Ejecuta: uv run python fillers.py")
        # Una bolsa barajada en vez de elegir al azar cada vez: con random.choice y seis
        # clips, la misma muletilla sale dos veces seguidas una de cada seis veces y tres de
        # cada cinco turnos repiten alguna. Barajar garantiza oirlas todas antes de repetir
        # ninguna, que es lo que hace que no suene a maquina.
        self._bolsas = {False: [], True: []}

    def pick(self, slow: bool = False):
        bolsa = self._bolsas[slow]
        if not bolsa:
            bolsa.extend(self.slow if slow else self.normal)
            random.shuffle(bolsa)
        return bolsa.pop()


if __name__ == "__main__":
    import sys

    from config import DEFAULT_PROFILE, load_config
    from tts import TTS

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROFILE)
    # Mismo ritmo y pausa que el robot: una muletilla que suena a otra velocidad delata
    # que es un clip grabado y no el mismo que esta hablando.
    tts = TTS(cfg.tts_engine, cfg.voice, cfg.tts_device,
              getattr(cfg, "voice_pace", 1.0), getattr(cfg, "sentence_pause_s", 0.0))
    out = FILLER_DIR / cfg.voice
    out.mkdir(parents=True, exist_ok=True)
    for viejo in out.glob("*.wav"):
        viejo.unlink()                 # si cambian las frases, los clips viejos no deben quedar
    for prefix, texts in (("filler", FILLERS), ("slow", SLOW_FILLERS)):
        for i, text in enumerate(texts):
            audio = tts.synthesize(text)
            save_wav(out / f"{prefix}_{i}.wav", audio, tts.sample_rate)
            print(f"{prefix}_{i}.wav  {len(audio) / tts.sample_rate:4.1f}s  {text}")
