from __future__ import annotations

import torch

from hooks.vo_hooks import GPT2VOActivationCollector


def collect_head_activations(adapter, texts: list[str], max_length: int, device: str) -> dict:
    tokenized = adapter.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    collector = GPT2VOActivationCollector(adapter)
    collector.register()
    adapter.model.eval()
    with torch.no_grad():
        adapter.model(**tokenized)
    collector.remove()

    return {
        "metadata": {
            "num_layers": adapter.num_layers,
            "num_heads": adapter.num_heads,
            "head_dim": adapter.head_dim,
            "num_sequences": len(texts),
        },
        "activations": collector.stacked(),
    }
