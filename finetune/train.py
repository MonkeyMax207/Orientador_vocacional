"""Full fine-tune of the small student model on the distilled dataset.

Runs on the DGX, in its own venv — never from the project's .venv, because the Pi installs
from pyproject.toml and must never pull torch:

    ~/.venvs/orienta-train/bin/python -m finetune.train --epochs 4
    ~/.venvs/orienta-train/bin/python -m finetune.train --dump-template

Full fine-tuning, not LoRA: a low-rank update is good at style and poor at memorizing new
facts, and a 0.6B with AdamW needs ~10 GB on a machine with 121 GB of unified memory.

The loss is masked to the assistant's tokens only. Without that the model learns to write
the student's questions too — the most common and most silent failure in SFT, so the masking
is computed here by hand instead of trusting a flag.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments)

from finetune import masking, memory

ROOT = Path(__file__).parent.parent
IGNORE = masking.IGNORE   # transformers' "do not compute loss here"

# Fixed messages whose rendering is the contract between training and Ollama (see Task 5).
FIXED_MESSAGES = [{"role": "system", "content": "SISTEMA"},
                  {"role": "user", "content": "USUARIO"},
                  {"role": "assistant", "content": "ASISTENTE"}]


# Qwen3's own template keeps the thinking scaffolding even with enable_thinking=False: it
# emits an empty "<think>\n\n</think>" before the answer. A robot that speaks every token it
# writes must never learn to produce markup, so training uses this minimal template instead —
# same <|im_start|>/<|im_end|> structure the model knows, without the reasoning block. The
# Ollama Modelfile mirrors it byte for byte (see finetune/TEMPLATE.md).
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def render(tokenizer, messages, add_generation_prompt=False) -> str:
    """One place decides how messages become text, so training and serving cannot drift."""
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=add_generation_prompt,
                                          chat_template=CHAT_TEMPLATE)


class ChatDataset(Dataset):
    """Tokenized conversations, with the loss masked to the assistant turns by finetune.masking.

    Every assistant turn of a dialogue carries labels, not just the last one: an 8-turn
    conversation is eight training signals.
    """

    def __init__(self, path: Path, tokenizer, max_len: int, excluir: tuple[str, ...] = ()):
        self.rows, self.tok, self.max_len = [], tokenizer, max_len
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if any(row.get("bucket", "").startswith(p) for p in excluir):
                continue
            self.rows.append(row["messages"])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        ids, labels = masking.build(
            self.rows[i],
            lambda msgs, gen: render(self.tok, msgs, add_generation_prompt=gen),
            lambda text: self.tok(text, add_special_tokens=False)["input_ids"])
        return {"input_ids": ids[:self.max_len] or [0],
                "labels": labels[:self.max_len] or [IGNORE]}


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad_id] * (width - len(b["input_ids"]))
                                    for b in batch]),
        "labels": torch.tensor([b["labels"] + [IGNORE] * (width - len(b["labels"]))
                                 for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (width - len(b["input_ids"]))
                                         for b in batch]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--data", default="finetune/data/train.jsonl")
    ap.add_argument("--out", default=str(Path.home() / "orienta-ft"))
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ga", type=int, default=2, help="acumulacion de gradiente; bs*ga = lote efectivo")
    ap.add_argument("--necesita-gb", type=float, default=memory.NEEDED_TRAIN_GB,
                    help="memoria que exige la compuerta previa; el 0.6B uso ~27 GB, un 1.7B pide mas")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--dump-template", action="store_true")
    ap.add_argument("--excluir", nargs="*", default=[],
                    help="prefijos de bucket a dejar fuera, p.ej. neg:fuera")
    ap.add_argument("--reanudar", nargs="?", const="auto", default=None,
                    help="retoma desde un checkpoint: ruta, o sin valor para el ultimo de --out")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.dump_template:
        rendered = render(tokenizer, FIXED_MESSAGES)
        (ROOT / "finetune" / "TEMPLATE.md").write_text(
            "# Plantilla de chat del entrenamiento\n\n"
            "Contrato entre `finetune/train.py` y el `Modelfile` de Ollama. Si estas dos cadenas\n"
            "difieren en un solo token, la calidad se cae sin ningun error visible.\n\n"
            f"Modelo: `{args.model}`  ·  `enable_thinking=False`\n\n"
            "Mensajes de prueba: " + json.dumps(FIXED_MESSAGES, ensure_ascii=False) + "\n\n"
            "```\n" + rendered + "\n```\n")
        print(rendered)
        return

    # Compuerta de memoria antes de reservar nada: el 2026-11-15 la maquina se congelo por OOM
    # global con dos instancias de Ollama reteniendo modelos mientras esto entrenaba.
    libre = memory.require(args.necesita_gb)
    print(f"memoria disponible: {libre:.1f} GB", flush=True)

    data = ChatDataset(ROOT / args.data, tokenizer, args.max_len, tuple(args.excluir))
    print(f"{len(data)} conversaciones (excluyendo {args.excluir or 'nada'})", flush=True)

    # Guardia: si la plantilla y el tokenizador no encajan, masking.build deja todo enmascarado
    # y el entrenamiento corre horas aprendiendo nada. Mejor abortar en el segundo cero.
    fracciones = [masking.trained_fraction(data[i]["labels"]) for i in range(min(20, len(data)))]
    media = sum(fracciones) / len(fracciones)
    print(f"fraccion de tokens con etiqueta: {media:.1%}", flush=True)
    # Umbral bajo a proposito: con el system prompt de la persona (~400 tokens) y respuestas de
    # ~20, un 5% etiquetado es lo normal y no indica fallo. Esto solo atrapa el caso de cero.
    if media < 0.02:
        raise SystemExit("El enmascarado no esta etiquetando nada: revisa CHAT_TEMPLATE.")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # transformers 5.x quito warmup_ratio: se calcula en pasos, y en una corrida corta (piloto)
    # un valor fijo seria mas largo que el entrenamiento entero.
    pasos_totales = max(1, int(len(data) / (args.bs * args.ga) * args.epochs))
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.ga,
            warmup_steps=max(1, int(pasos_totales * 0.03)), lr_scheduler_type="cosine",
            bf16=True, logging_steps=20, report_to=[],
            # Cada 200 pasos y no por epoca: el crash del 2026-11-15 se llevo una hora de
            # entrenamiento porque el primer checkpoint de epoca no habia alcanzado a escribirse.
            save_strategy="steps", save_steps=200, save_total_limit=4,
        ),
        train_dataset=data,
        data_collator=lambda b: collate(b, pad_id),
    )
    # Reanudar importa tanto como entrenar: el 2026-11-16 un corte de luz mato la corrida en el
    # paso 2155 de 4075 con checkpoints perfectos en disco, y sin esta bandera la unica salida
    # era repetir dos horas y cuarto de GPU. El checkpoint trae pesos, momentos del optimizador,
    # posicion del scheduler y estado del RNG, asi que retomar no es empezar de nuevo mas rapido:
    # es la misma corrida continuada.
    desde = args.reanudar
    if desde == "auto":
        puntos = sorted(Path(args.out).glob("checkpoint-*"),
                        key=lambda d: int(d.name.split("-")[1]))
        if not puntos:
            raise SystemExit(f"No hay checkpoint-* en {args.out} para reanudar.")
        desde = str(puntos[-1])
        print(f"reanudando desde {desde}", flush=True)
    trainer.train(resume_from_checkpoint=desde)
    trainer.save_model(args.out + "/final")
    tokenizer.save_pretrained(args.out + "/final")
    print(f"listo -> {args.out}/final", flush=True)


if __name__ == "__main__":
    main()
