from __future__ import annotations

from collections.abc import Sequence


def get_text_splits(config: dict) -> tuple[list[str], list[str]]:
    splits = load_named_splits(config)
    calibration_key = config.get("calibration_split", "S0_local")
    evaluation_key = config.get("evaluation_split", "S3_test")
    if calibration_key in splits and evaluation_key in splits:
        return list(splits[calibration_key]), list(splits[evaluation_key])

    texts = list(config["texts"])
    calibration_count = int(config.get("calibration_texts", len(texts)))
    evaluation_count = int(config.get("evaluation_texts", len(texts)))
    return texts[:calibration_count], texts[:evaluation_count]


def load_named_splits(config: dict) -> dict[str, list[str]]:
    if "named_splits" in config:
        return {key: list(value) for key, value in config["named_splits"].items()}
    if "dataset" in config:
        return _load_dataset_splits(config["dataset"])
    return {}


def _load_dataset_splits(dataset_config: dict) -> dict[str, list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_config["path"], dataset_config.get("name"))
    text_field = dataset_config.get("text_field", "text")
    min_chars = int(dataset_config.get("min_chars", 1))

    def select_examples(split_name: str, start: int, count: int) -> list[str]:
        source = dataset[split_name]
        collected: list[str] = []
        cursor = int(start)
        while len(collected) < count and cursor < len(source):
            text = str(source[cursor][text_field]).strip()
            if len(text) >= min_chars:
                collected.append(text)
            cursor += 1
        if len(collected) < count:
            raise ValueError(
                f"requested {count} usable texts from split '{split_name}', got {len(collected)}"
            )
        return collected

    return {
        split_name: select_examples(
            split_spec["hf_split"],
            int(split_spec.get("start", 0)),
            int(split_spec["count"]),
        )
        for split_name, split_spec in dataset_config["splits"].items()
    }


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
        padding: bool | str = True,
    ):
        import torch

        del padding
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
            padding_ids = [self.pad_token_id] * (max_seq - len(seq))
            input_ids.append(seq + padding_ids)
            attention_mask.append([1] * len(seq) + [0] * len(padding_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
