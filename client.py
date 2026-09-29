"""Pi side: ears and mouth. Detects the user's turn locally (Silero VAD, cheap), says "Jum..." at once,
sends the turn to the PC over a WebSocket and plays each answer sentence as it arrives.

Run on the Pi:  uv run python client.py        (server_url in config.toml → the PC's address)
"""
import asyncio
import json
import sys

import numpy as np
import websockets

from audio_io import Mic, Player, to_int16
from config import load_config
from vad import VAD, TurnDetector


async def main(cfg):
    async with websockets.connect(cfg.server_url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        await ws.recv()   # the "Jum..." filler clip: received to stay in sync with the protocol, but
                          # not played (disabled until natural-sounding fillers are recorded — see §4)
        # voice_rate < 1 plays the same samples slower: deeper "turtle" voice at zero CPU cost.
        player = Player(cfg.output_device, int(hello["sample_rate"] * cfg.voice_rate))
        mic, vad = Mic(cfg.input_device), VAD(cfg.vad_threshold)
        turns = TurnDetector(cfg.silence_ms, cfg.min_speech_ms)
        print(f"Conectado a {cfg.server_url}. Habla cuando quieras. Ctrl+C para salir.", flush=True)

        def listen():   # blocking loop over 32 ms mic chunks until a full turn is detected
            while True:
                chunk = mic.read()
                audio = turns.feed(chunk, vad.is_speech(chunk))
                if audio is not None:
                    return audio

        while True:
            audio = await asyncio.to_thread(listen)   # keep the WebSocket alive while listening
            await ws.send(to_int16(audio).tobytes())
            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "done":
                    break
                if msg["type"] == "heard":
                    print(f"\nTú: {msg['text']}", flush=True)
                elif msg["type"] == "say":
                    print(f"Orienta: {msg['text']}", flush=True)
                    player.put(np.frombuffer(await ws.recv(), dtype=np.int16), msg["anims"])
            while player.busy():                      # wait until it finishes speaking...
                await asyncio.sleep(0.03)
            mic.pending()                             # ...then drop what the mic heard meanwhile (its own voice)
            vad.reset()
            turns.reset()


if __name__ == "__main__":
    try:
        asyncio.run(main(load_config(sys.argv[1] if len(sys.argv) > 1 else "pi")))
    except KeyboardInterrupt:
        print("\nAdiós")
