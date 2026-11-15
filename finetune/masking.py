"""Which tokens carry the loss: the assistant's, and nothing else.

Isolated from train.py (and from torch) because this is the correctness heart of the
training and the most silent way to get it wrong — a model trained without this learns to
write the student's questions as well as the answers.

Pure functions over two injected callables, so the logic is testable without a tokenizer:
  render(messages, add_generation_prompt) -> str
  encode(text) -> list[int]
"""
IGNORE = -100


def build(messages: list[dict], render, encode) -> tuple[list[int], list[int]]:
    """Token ids for the whole conversation, and labels masked to every assistant span.

    Each assistant message's span is the difference between encoding the conversation up to
    it (with the generation prompt) and up to and including it. Chat templates are
    append-only, so a prefix's encoding is a prefix of the whole — verified per span, and a
    span that fails the check is left masked rather than guessed.
    """
    ids = encode(render(messages, False))
    labels = [IGNORE] * len(ids)
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix = encode(render(messages[:i], True))
        upto = encode(render(messages[:i + 1], False))
        start, end = len(prefix), min(len(upto), len(ids))
        if start >= end or ids[:start] != prefix:
            continue
        labels[start:end] = ids[start:end]
    return ids, labels


def trained_fraction(labels: list[int]) -> float:
    """Share of tokens that carry a label. A dataset where this is ~0 trains nothing."""
    return sum(1 for l in labels if l != IGNORE) / len(labels) if labels else 0.0
