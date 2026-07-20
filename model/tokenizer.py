"""Character-level tokenizer with BOS (Karpathy microgpt style)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharTokenizer:
    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.bos_id = len(self.chars)
        self.vocab_size = len(self.chars) + 1

    @classmethod
    def from_texts(cls, texts: list[str]) -> CharTokenizer:
        chars = sorted(set("".join(texts)))
        return cls(chars=chars)

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        return cls(chars=list(data["chars"]))

    def to_dict(self) -> dict:
        return {"chars": self.chars}

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        if add_bos:
            return [self.bos_id] + ids + [self.bos_id]
        return ids

    def encode_prompt(self, text: str) -> list[int]:
        """BOS + characters, no trailing BOS (for continuation)."""
        ids = [self.stoi[ch] for ch in text if ch in self.stoi]
        return [self.bos_id] + ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == self.bos_id:
                continue
            if i in self.itos:
                out.append(self.itos[i])
        return "".join(out)
