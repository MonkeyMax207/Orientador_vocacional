"""Turns the LLM's token stream into speakable sentences.

Why: Piper needs whole sentences to sound natural, but waiting for the whole reply
wastes seconds. So we speak each sentence the moment it is complete.
"""
import re

# Words that end with a dot but do NOT end a sentence (compared in lowercase).
ABBREVIATIONS = {"dr", "dra", "sr", "sra", "srta", "ud", "uds", "etc", "ej", "pág", "núm", "aprox", "lic", "ing"}

# A sentence end = one or more of . ! ? … (optionally followed by a closing quote or
# parenthesis), and then whitespace. Requiring the whitespace is what keeps "3.5" together,
# and it means we only decide once we have seen the character after the punctuation.
_BOUNDARY = re.compile(r'([.!?…]+["»)]?)\s+')


class SentenceSplitter:
    def __init__(self):
        self.buf = ""   # text received but not yet returned as a sentence

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out, start = [], 0
        for m in _BOUNDARY.finditer(self.buf):
            candidate = self.buf[start:m.end(1)]   # from the last cut up to the punctuation
            if m.group(1) == ".":
                last_word = re.search(r"(\w+)\.$", candidate)
                if last_word and last_word.group(1).lower() in ABBREVIATIONS:
                    continue                        # "Dr." → keep reading, not an end
            out.append(candidate.strip())
            start = m.end()                         # next sentence starts after the whitespace
        self.buf = self.buf[start:]                 # keep the unfinished tail for the next feed
        return [s for s in out if s]

    def flush(self) -> list[str]:
        # Called when the LLM stream ends: whatever is left is the last sentence.
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


# Characters the voice should never read: markdown symbols...
_MARKDOWN = re.compile(r"[*_#`~>|]")
# ...emojis and pictographs (Unicode blocks), plus the invisible emoji variation selector...
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF☀-➿️]")
# ...and list bullets at the start of a line.
_BULLET = re.compile(r"^\s*[-•]\s+")


def clean_for_tts(text: str) -> str:
    text = _BULLET.sub("", text)
    text = _EMOJI.sub("", _MARKDOWN.sub("", text))
    return re.sub(r"\s+", " ", text).strip()   # collapse the spaces left behind
