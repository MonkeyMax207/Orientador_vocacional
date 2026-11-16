"""Thermal guard for long GPU runs on the DGX: freeze the heavy work before the box cooks.

Adapted from animation_creator/orchestrator/thermal.py, which froze docker containers labelled
genesis.heavy. Here the heavy work is bare processes (the Ollama runner serving the teacher, and
the fine-tune), so it freezes PIDs with SIGSTOP/SIGCONT instead — the same mechanism docker
pause uses underneath (cgroup freezer), and safe mid-CUDA: queued GPU work drains, then idles.

Thresholds come from that project's measurements (2026-09-27, Wan 14B render): hottest ACPI zone
95 °C with GPU at 82 °C and SW thermal slowdown active; every zone's critical trip point, where
the machine powers off, is 104 °C.

  >= 90 °C  before starting new work: wait_until_cool() holds until <= 85 °C
  >= 94 °C  while working: SIGSTOP the heavy PIDs; resume at <= 85 °C. Nothing is lost —
            dataset generation resumes by job id and training checkpoints every epoch.
  >= 100 °C SIGKILL them (last resort; shutdown at 104 °C)

Run detached beside a long job:
    uv run python -m finetune.thermal
"""
import glob
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from finetune import memory

ZONES = "/sys/class/thermal/thermal_zone*/temp"
KILL_AT_C = 100.0
PAUSE_AT_C = 94.0
START_AT_C = 90.0
RESUME_AT_C = 85.0
LOG_EVERY_S = 30
# The processes that actually heat the GPU. make_dataset only drives HTTP, so freezing it would
# cool nothing: it is the Ollama runner that holds the model.
HEAVY = ("ollama runner", "finetune.train")
LOG_PATH = Path(__file__).parent.parent / "logs" / "thermal.csv"


def read_zone_temps(pattern: str = ZONES) -> list[float]:
    temps = []
    for path in sorted(glob.glob(pattern)):
        try:
            temps.append(int(Path(path).read_text().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return temps


def heavy_pids(patterns: tuple[str, ...] = HEAVY) -> list[int]:
    """PIDs whose command line matches a heavy pattern, read straight from /proc.

    Deliberately not pgrep/pkill: a pattern passed on a command line matches that very command,
    which is how a careless pkill kills its own caller.
    """
    mine = {os.getpid(), os.getppid()}
    found = []
    for entry in glob.glob("/proc/[0-9]*/cmdline"):
        pid = int(entry.split("/")[2])
        if pid in mine:
            continue
        try:
            cmdline = Path(entry).read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if any(p in cmdline for p in patterns):
            found.append(pid)
    return sorted(found)


def _signal_all(pids: list[int], sig: int) -> list[int]:
    sent = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent.append(pid)
        except OSError:
            pass
    return sent


def log(message: str, path: Path = LOG_PATH) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks_event_reasons.active",
                                "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


class ThermalGuard:
    """Freezes and thaws the heavy processes by temperature. Injectable for tests."""

    def __init__(self, read=read_zone_temps, pids=heavy_pids, send=_signal_all, log=log,
                 pause_at: float = PAUSE_AT_C, resume_at: float = RESUME_AT_C,
                 kill_at: float = KILL_AT_C):
        self.read, self.pids, self.send, self.log = read, pids, send, log
        self.pause_at, self.resume_at, self.kill_at = pause_at, resume_at, kill_at
        self.frozen: list[int] = []

    def tick(self, pressure: bool = False) -> float:
        """`pressure` is a second reason to freeze (memory), judged by the caller.

        One state machine for both hazards: the response to heat and to memory exhaustion is the
        same — freeze the heavy work — and two daemons tracking the same frozen PIDs would race.
        """
        temps = self.read()
        hottest = max(temps) if temps else 0.0
        if pressure and not self.frozen:
            self.frozen = self.send(self.pids(), signal.SIGSTOP)
            self.log(f"MEMORIA PAUSE {self.frozen or '(nada pesado)'}: poca memoria disponible")
            return hottest
        if pressure:                              # sigue congelado hasta que ceda la presion
            return hottest
        if hottest >= self.kill_at:
            killed = self.send(self.frozen or self.pids(), signal.SIGKILL)
            self.frozen = []
            self.log(f"THERMAL KILL {killed or '(nada pesado)'}: {hottest:.1f} °C >= "
                     f"{self.kill_at:.0f} °C (apagado a 104 °C)")
        elif not self.frozen and hottest >= self.pause_at:
            self.frozen = self.send(self.pids(), signal.SIGSTOP)
            self.log(f"THERMAL PAUSE {self.frozen or '(nada pesado)'}: {hottest:.1f} °C >= "
                     f"{self.pause_at:.0f} °C; congelado hasta <= {self.resume_at:.0f} °C")
        elif self.frozen and hottest <= self.resume_at:
            self.send(self.frozen, signal.SIGCONT)
            self.log(f"THERMAL RESUME {self.frozen}: {hottest:.1f} °C")
            self.frozen = []
        return hottest


def wait_until_cool(read=read_zone_temps, sleep=time.sleep, start_at: float = START_AT_C,
                    resume_at: float = RESUME_AT_C, max_wait_s: float = 3600,
                    now=time.time, log=log) -> bool:
    """Holds before starting heavy work. True if it had to wait; TimeoutError if cooling fails."""
    hottest = max(read() or [0.0])
    if hottest < start_at:
        return False
    log(f"ESPERANDO A QUE ENFRIE: {hottest:.1f} °C >= {start_at:.0f} °C, hasta <= {resume_at:.0f} °C")
    deadline = now() + max_wait_s
    while hottest > resume_at:
        if now() > deadline:
            raise TimeoutError(f"sigue a {hottest:.1f} °C tras {max_wait_s:.0f} s: revisa la ventilacion")
        sleep(30)
        hottest = max(read() or [0.0])
    log(f"ENFRIADO: {hottest:.1f} °C")
    return True


def main() -> None:
    guard = ThermalGuard()
    log(f"guardia activa: pausa {PAUSE_AT_C:.0f} °C, reanuda {RESUME_AT_C:.0f} °C, "
        f"mata {KILL_AT_C:.0f} °C, piso de memoria {memory.FLOOR_GB:.0f} GB; vigilando {HEAVY}")
    last_csv = 0.0
    while True:
        try:
            libre = memory.available_gb()
            hottest = guard.tick(pressure=libre < memory.FLOOR_GB)
            if time.time() - last_csv >= LOG_EVERY_S:
                last_csv = time.time()
                log(f"{hottest:.1f} °C  {libre:.1f} GB libres  gpu[{gpu_state()}]  "
                    f"{'CONGELADO ' + str(guard.frozen) if guard.frozen else ''}")
        except Exception as e:                      # nunca morir: es el guardia
            log(f"error en la guardia: {e!r}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
