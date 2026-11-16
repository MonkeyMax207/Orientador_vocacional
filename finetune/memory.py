"""Memory headroom checks for the DGX Spark, where CPU and GPU share one pool.

Adapted from animation_creator/orchestrator/memguard.py. Its header states the thing that
matters here: on this machine "available memory" in /proc/meminfo is the number that decides
whether a model fits, because there is no separate VRAM.

Why this exists at all: on 2026-11-15 the box froze under a global OOM at 19:16 while two Ollama
instances (both with keep_alive=-1, so models never unloaded) and a fine-tune competed for the
121 GB pool. Nothing was attributable — GPU allocations do not show up as process RSS, and
nvidia-smi reports memory.used as [N/A] here — so the OOM killer executed desktop daemons worth
a few hundred KB each and the session locked up. Ollama's own log shows why it never held back:

    NVML not supported for memory query, using system memory (total=130662940672, ...)

It budgets against the whole machine. So the caller has to check, before starting and while
running.
"""
from pathlib import Path

KB_PER_GB = 1024 ** 2
# A full fine-tune of a 0.6B with AdamW is ~10 GB of weights/grads/optimizer plus activations.
NEEDED_TRAIN_GB = 25.0
# Left for the desktop, the page cache and the kernel. Below this the machine stops responding
# before anything useful gets killed.
RESERVE_GB = 8.0
# While running: under this much available, freeze the heavy work.
FLOOR_GB = 12.0


def meminfo(path: str = "/proc/meminfo") -> dict[str, float]:
    """MemTotal and MemAvailable in GB."""
    values = {}
    for line in Path(path).read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) / KB_PER_GB
    return {"total_gb": values.get("MemTotal", 0.0),
            "available_gb": values.get("MemAvailable", 0.0)}


def available_gb(path: str = "/proc/meminfo") -> float:
    return meminfo(path)["available_gb"]


def check(needed_gb: float = NEEDED_TRAIN_GB, reserve_gb: float = RESERVE_GB,
          path: str = "/proc/meminfo") -> tuple[bool, str]:
    """(ok, message). Separated from require() so tests need no exceptions."""
    free = available_gb(path)
    required = needed_gb + reserve_gb
    if free < required:
        return False, (f"hacen falta {required:.0f} GB disponibles ({needed_gb:.0f} de trabajo + "
                       f"{reserve_gb:.0f} de reserva) y solo hay {free:.1f} GB. Descarga los "
                       f"modelos de Ollama antes de entrenar: 'ollama stop <modelo>' o "
                       f"'systemctl restart ollama'.")
    return True, f"{free:.1f} GB disponibles, hacen falta {required:.0f}"


def require(needed_gb: float = NEEDED_TRAIN_GB, reserve_gb: float = RESERVE_GB,
            path: str = "/proc/meminfo") -> float:
    """Pre-flight gate: aborts rather than starting work that will OOM the machine."""
    ok, message = check(needed_gb, reserve_gb, path)
    if not ok:
        raise SystemExit(f"MEMORIA INSUFICIENTE: {message}")
    return available_gb(path)
