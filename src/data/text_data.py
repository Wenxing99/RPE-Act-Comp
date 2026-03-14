from __future__ import annotations

from collections.abc import Sequence


def get_text_splits(config: dict) -> tuple[list[str], list[str]]:
    texts = list(config["texts"])
    calibration_count = int(config.get("calibration_texts", len(texts)))
    evaluation_count = int(config.get("evaluation_texts", len(texts)))
    return texts[:calibration_count], texts[:evaluation_count]


class SimpleCharTokenizer:
    def __init__(self, vocab_size: int = 256) -> None:
        if vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str, max_length: int) -> list[int]:
        token_ids = [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
        token_ids = token_ids[: max_length - 1]
        token_ids.append(self.eos_token_id)
        return token_ids

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int,
    ):
        import torch

        if return_tensors != "pt":
            raise ValueError("SimpleCharTokenizer only supports return_tensors='pt'")
        if isinstance(texts, str):
            texts = [texts]

        sequences = []
        for text in texts:
            token_ids = self.encode(text, max_length=max_length if truncation else 1000000)
            if truncation:
                token_ids = token_ids[:max_length]
            sequences.append(token_ids)

        max_seq = max(len(seq) for seq in sequences)
        input_ids = []
        attention_mask = []
        for seq in sequences:
            padding = [self.pad_token_id] * (max_seq - len(seq))
            input_ids.append(seq + padding)
            attention_mask.append([1] * len(seq) + [0] * len(padding))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
