"""Prints every audio device and checks that the ones named in config.toml work.

Run: uv run python list_devices.py [pc|pi]
"""
import sys

import sounddevice as sd

from config import load_config

# Each line: index, name, host API (MME / DirectSound / WASAPI on Windows, ALSA on Linux)
# and "(in, out)" channel counts. '>' marks the default input, '<' the default output.
print(sd.query_devices())

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "pc")
for kind, name in (("input", cfg.input_device), ("output", cfg.output_device)):
    try:
        # sounddevice matches the words in `name` against "<device name> (<host API>)".
        if kind == "input":
            sd.check_input_settings(device=name, samplerate=16000, channels=1, dtype="float32")
        else:
            sd.check_output_settings(device=name, channels=1, dtype="int16")
        print(f"OK   {kind}: '{name}'")
    except Exception as e:  # ValueError if not found or ambiguous; PortAudioError if unsupported
        print(f"FAIL {kind}: '{name}' → {e}\n     Edit input_device/output_device in config.toml")
