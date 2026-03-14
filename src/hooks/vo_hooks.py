from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class GPT2VOActivationCollector:
    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self.handles: list[Any] = []
        self.value_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)
        self.output_by_layer: dict[int, list[torch.Tensor]] = defaultdict(list)

    def register(self) -> None:
        for layer_index, layer in enumerate(self.adapter.get_attention_layers()):
            self.handles.append(
                layer.c_attn.register_forward_hook(self._make_value_hook(layer_index))
            )
            self.handles.append(
                layer.c_proj.register_forward_pre_hook(self._make_output_hook(layer_index))
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_value_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            self.value_by_layer[layer_index].append(
                self.adapter.extract_value_heads(output.detach()).cpu()
            )

        return hook

    def _make_output_hook(self, layer_index: int):
        def hook(_module, inputs):
            self.output_by_layer[layer_index].append(
                self.adapter.extract_output_heads(inputs[0].detach()).cpu()
            )

        return hook

    def stacked(self) -> dict[str, dict[int, torch.Tensor]]:
        return {
            "value": {
                layer_index: torch.cat(chunks, dim=0)
                for layer_index, chunks in self.value_by_layer.items()
            },
            "output": {
                layer_index: torch.cat(chunks, dim=0)
                for layer_index, chunks in self.output_by_layer.items()
            },
        }
