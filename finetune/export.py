"""Export the fine-tuned model into Ollama, which is what the Pi actually runs.

Why Ollama and not llama.cpp's quantize: only `llama-server` was built here, and Ollama is the
deployment runtime anyway. Quantizing in the same engine that will serve it means the number we
measure is the number the robot gets, instead of a figure from a different stack.

The Modelfile is generated, never hand-written, for one reason: the chat template and the system
prompt must match training byte for byte. `finetune/train.py` renders with CHAT_TEMPLATE and the
dataset was built from `brain.SYSTEM_PROMPT`, so both are imported here rather than retyped. A
template that differs by one token degrades quality with no error anywhere — the quietest
failure in this whole project, and the reason save_pretrained's own template had to be
overwritten on 2026-11-16.

    uv run python -m finetune.export --nombre orienta-17b --cuantizar q4_K_M
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Go template for Ollama, equivalent to train.py's Jinja CHAT_TEMPLATE:
#   {% for m in messages %}<|im_start|>{{role}}\n{{content}}<|im_end|>\n{% endfor %}
#   {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
# Written out rather than translated at runtime, and verified by --verificar, which renders
# both and compares them character by character.
PLANTILLA_GO = """{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""


def modelfile(origen: Path, system: str) -> str:
    """The Modelfile text. `origen` is the HF directory; Ollama imports safetensors directly."""
    return (
        f"FROM {origen}\n\n"
        f'TEMPLATE """{PLANTILLA_GO}"""\n\n'
        f'SYSTEM """{system}"""\n\n'
        "# El robot habla: nada de markup, respuestas cortas, y para en <|im_end|>.\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
        "PARAMETER temperature 0.3\n"
        "PARAMETER num_predict 120\n"
        # El contexto debe caber los 3 documentos de BM25 (~600 palabras) mas el system prompt.
        "PARAMETER num_ctx 4096\n"
    )


def render_jinja(messages: list[dict]) -> str:
    """What training produced, via the same function training used."""
    from transformers import AutoTokenizer

    from finetune.train import render
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    return render(tok, messages, add_generation_prompt=True)


def render_go(messages: list[dict]) -> str:
    """What Ollama will produce from PLANTILLA_GO, simulated.

    Ollama's Go template is not available to Python, so this reimplements exactly what the
    template above does — a loop over the messages and the assistant opening. If the two
    renderings disagree the export is wrong, which is the whole point of checking.
    """
    partes = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(partes) + "<|im_start|>assistant\n"


def verificar() -> bool:
    """True when both renderings are identical. Prints the first difference otherwise."""
    casos = [
        [{"role": "system", "content": "SISTEMA"}, {"role": "user", "content": "USUARIO"}],
        [{"role": "system", "content": "S"}, {"role": "user", "content": "u1"},
         {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}],
    ]
    ok = True
    for i, msgs in enumerate(casos, 1):
        a, b = render_jinja(msgs), render_go(msgs)
        if a == b:
            print(f"caso {i}: identico ({len(a)} caracteres)")
            continue
        ok = False
        print(f"caso {i}: DIFIEREN")
        for j, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  primera diferencia en el caracter {j}: {x!r} vs {y!r}")
                print(f"  entrenamiento: {a[max(0, j-40):j+40]!r}")
                print(f"  ollama:        {b[max(0, j-40):j+40]!r}")
                break
        else:
            print(f"  uno es prefijo del otro: {len(a)} vs {len(b)} caracteres")
    return ok


STORE = Path("/usr/share/ollama/.ollama/models")


def empaquetar(nombre: str, destino: Path) -> Path:
    """Saca el GGUF ya cuantizado del almacen de Ollama a un archivo con nombre.

    Ollama guarda cada capa como un blob con nombre de hash; la capa "model" ES el GGUF. Se copia
    en vez de reconvertir para que la Pi reciba exactamente los bytes que se midieron aqui: una
    cuantizacion repetida en otra maquina no tiene por que dar el mismo archivo, y entonces la
    cifra medida dejaria de corresponder al modelo desplegado.
    """
    import shutil

    manifiesto = STORE / "manifests" / "registry.ollama.ai" / "library" / nombre / "latest"
    if not manifiesto.exists():
        raise SystemExit(f"No existe el manifiesto de '{nombre}' en {manifiesto}")
    capas = json.loads(manifiesto.read_text())["layers"]
    modelo = next((c for c in capas if c["mediaType"].endswith(".model")), None)
    if not modelo:
        raise SystemExit(f"El manifiesto de '{nombre}' no tiene capa de modelo")
    blob = STORE / "blobs" / modelo["digest"].replace(":", "-")

    destino.mkdir(parents=True, exist_ok=True)
    gguf = destino / f"{nombre}.gguf"
    print(f"copiando {modelo['size'] / 1e9:.2f} GB -> {gguf}", flush=True)
    shutil.copy2(blob, gguf)

    from brain import SYSTEM_PROMPT
    # El Modelfile de la Pi apunta al GGUF que queda a su lado, con ruta relativa.
    texto = modelfile(Path(f"./{gguf.name}"), SYSTEM_PROMPT)
    (destino / "Modelfile").write_text(texto)
    print(f"Modelfile -> {destino / 'Modelfile'}")
    print(f"\nEn la Pi:\n  cd {destino.name} && ollama create {nombre} -f Modelfile")
    return gguf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", default=str(Path.home() / "orienta-ft-17b" / "final"))
    ap.add_argument("--nombre", default="orienta-17b")
    ap.add_argument("--cuantizar", default="", help="q4_K_M, q8_0, o vacio para f16")
    ap.add_argument("--verificar", action="store_true", help="solo comprobar la plantilla")
    ap.add_argument("--empaquetar", metavar="DESTINO",
                    help="copia el GGUF ya cuantizado de --nombre mas su Modelfile a DESTINO, "
                          "listo para llevar a la Pi en una USB")
    args = ap.parse_args()

    if args.empaquetar:
        empaquetar(args.nombre, Path(args.empaquetar))
        return

    if not verificar():
        raise SystemExit("La plantilla de Ollama NO coincide con la del entrenamiento. "
                         "Exportar asi degradaria la calidad sin ningun error visible.")
    if args.verificar:
        return

    from brain import SYSTEM_PROMPT
    destino = ROOT / "finetune" / "Modelfile"
    destino.write_text(modelfile(Path(args.origen), SYSTEM_PROMPT))
    print(f"Modelfile -> {destino}")

    cmd = ["ollama", "create", args.nombre, "-f", str(destino)]
    if args.cuantizar:
        cmd += ["--quantize", args.cuantizar]
    print("$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:] or r.stderr[-3000:])
    if r.returncode != 0:
        raise SystemExit(f"ollama create fallo con codigo {r.returncode}")
    print(f"listo: {args.nombre}")


if __name__ == "__main__":
    main()
