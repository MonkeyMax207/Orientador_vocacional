"""PC side: the brain. Receives a user turn (audio) from the Pi over a WebSocket and streams back
the spoken answer sentence by sentence (Kokoro audio + animation names).

Run on the PC:  uv run python server.py [pc]

Protocol (one WebSocket connection; text frames are JSON, binary frames are int16 mono audio):
  server → Pi on connect : {"type":"hello","sample_rate":24000}  then  <binary filler clip "Jum...">
  Pi → server per turn   : <binary 16 kHz audio of the user's turn>
  server → Pi per turn   : {"type":"heard","text":...}
                           {"type":"say","text":...,"anims":[...]}  then  <binary audio>   (repeated)
                           {"type":"done"}
"""
import asyncio
import json
import sys
import threading

import numpy as np
import websockets

import animations
from brain import make_brain
from config import load_config
from fillers import Fillers
from rag import make_finder
from sentences import SentenceSplitter, clean_for_tts
from stt import STT, clean
from tts import TTS


class Pipeline:
    """Everything that used to run in main.py after the microphone: STT → RAG → LLM → TTS."""

    def __init__(self, cfg):
        self.cfg = cfg
        # UAO text chunks go next to the student's message only for questions about the UAO.
        knowledge, self.find = make_finder(cfg)
        names = knowledge.program_names() if knowledge else ()
        self.brain = make_brain(cfg, program_names=names)
        self.brain.check()
        self.brain.warmup()
        self.stt = STT(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute, cfg.threads)
        self.tts = TTS(cfg.tts_engine, cfg.voice, cfg.tts_device)
        self.tts.synthesize("Hola.")   # warm-up
        self.filler = Fillers(cfg.voice).pick()
        self.previous = ""   # the student's previous turn (topic context for the search)

    def answer(self, audio_bytes: bytes):
        """Generator: yields ("heard", text) once, then ("say", text, anims, int16 clip) per sentence."""
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768
        text = clean(self.stt.transcribe(audio))
        if not text:
            return   # noise: nothing to say
        yield ("heard", text)
        splitter = SentenceSplitter()
        cards = self.find(text, self.previous)
        self.previous = text
        if cards:
            print(f"[rag] {', '.join(c['id'] for c in cards)}", flush=True)
        for token in [*self.brain.stream_reply(text, cards=cards), None]:
            for sentence in splitter.flush() if token is None else splitter.feed(token):
                sentence = clean_for_tts(sentence)
                if sentence:
                    yield ("say", sentence, animations.match(sentence), self.tts.synthesize(sentence))


async def serve(pipeline: Pipeline, port: int):
    async def handle(ws):
        print("Pi conectada", flush=True)
        await ws.send(json.dumps({"type": "hello", "sample_rate": pipeline.tts.sample_rate}))
        await ws.send(pipeline.filler.tobytes())
        loop = asyncio.get_running_loop()
        async for msg in ws:                      # each binary message = one user turn
            q = asyncio.Queue()

            def work():   # the pipeline blocks (GPU work): run it on a thread, hand results to asyncio
                try:
                    for item in pipeline.answer(msg):
                        loop.call_soon_threadsafe(q.put_nowait, item)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            threading.Thread(target=work, daemon=True).start()
            while (item := await q.get()) is not None:
                if item[0] == "heard":
                    print(f"\nTú: {item[1]}", flush=True)
                    await ws.send(json.dumps({"type": "heard", "text": item[1]}))
                else:
                    _, text, anims, clip = item
                    print(f"Orienta: {text}", flush=True)
                    await ws.send(json.dumps({"type": "say", "text": text, "anims": anims}))
                    await ws.send(clip.tobytes())   # sent as soon as each sentence is ready
            await ws.send(json.dumps({"type": "done"}))

    # 0.0.0.0 = accept connections from other machines on the network (the Pi), not only this PC.
    async with websockets.serve(handle, "0.0.0.0", port, max_size=None):
        print(f"Servidor listo en el puerto {port}. Esperando a la Pi...", flush=True)
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
    asyncio.run(serve(Pipeline(cfg), cfg.server_port))
