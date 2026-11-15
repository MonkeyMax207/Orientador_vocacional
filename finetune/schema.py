"""Training example format.

The system prompt is imported from brain.py, never copied: the student must be trained on
byte-identical text to what it will be served. brain.Brain already drops RAG_RULES and the
program list when there is no knowledge object, which is exactly the Pi's configuration,
so SYSTEM_PROMPT alone is what the fine-tuned model will see in production.
"""
from brain import SYSTEM_PROMPT


def to_chat(example: dict) -> dict:
    """A generated example as a chat record: {"messages": [{"role", "content"}, ...]}."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "turnos" in example:
        for turn in example["turnos"]:
            messages.append({"role": "user", "content": turn["u"]})
            messages.append({"role": "assistant", "content": turn["a"]})
    else:
        messages.append({"role": "user", "content": example["p"]})
        messages.append({"role": "assistant", "content": example["r"]})
    return {"messages": messages}
