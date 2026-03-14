from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from data.text_data import SimpleCharTokenizer


@dataclass
class CompressionSpec:
    layer_index: int
    rank: int
    bases: dict[int, torch.Tensor]


class CompressedGPT2Attention(nn.Module):
    def __init__(self, original, basis_by_head: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.original = original
        self.embed_dim = original.embed_dim
        self.num_heads = original.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.rank = next(iter(basis_by_head.values())).shape[1]
        self.basis_by_head = {
            head_index: basis.detach().clone() for head_index, basis in basis_by_head.items()
        }
        self.resid_dropout = deepcopy(original.resid_dropout)
        self.attn_dropout = deepcopy(original.attn_dropout)
        self.scale_attn_weights = original.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = original.scale_attn_by_inverse_layer_idx
        self.layer_idx = getattr(original, "layer_idx", 0)

        qkv_weight = original.c_attn.weight.detach().clone()
        qkv_bias = original.c_attn.bias.detach().clone()
        proj_weight = original.c_proj.weight.detach().clone()
        proj_bias = original.c_proj.bias.detach().clone()

        self.q_weight = nn.Parameter(qkv_weight[:, : self.embed_dim], requires_grad=False)
        self.k_weight = nn.Parameter(
            qkv_weight[:, self.embed_dim : 2 * self.embed_dim], requires_grad=False
        )
        self.q_bias = nn.Parameter(qkv_bias[: self.embed_dim], requires_grad=False)
        self.k_bias = nn.Parameter(
            qkv_bias[self.embed_dim : 2 * self.embed_dim], requires_grad=False
        )

        value_weight = qkv_weight[:, 2 * self.embed_dim :]
        value_bias = qkv_bias[2 * self.embed_dim :]

        low_rank_value_weights = []
        low_rank_value_biases = []
        low_rank_output_weights = []
        for head_index in range(self.num_heads):
            basis = self.basis_by_head[head_index]
            start = head_index * self.head_dim
            end = start + self.head_dim
            head_value_weight = value_weight[:, start:end]
            head_value_bias = value_bias[start:end]
            head_output_weight = proj_weight[start:end, :]
            low_rank_value_weights.append(head_value_weight @ basis)
            low_rank_value_biases.append(head_value_bias @ basis)
            low_rank_output_weights.append(basis.transpose(0, 1) @ head_output_weight)

        self.low_rank_value_weight = nn.Parameter(
            torch.cat(low_rank_value_weights, dim=1),
            requires_grad=False,
        )
        self.low_rank_value_bias = nn.Parameter(
            torch.cat(low_rank_value_biases, dim=0),
            requires_grad=False,
        )
        self.low_rank_output_weight = nn.Parameter(
            torch.cat(low_rank_output_weights, dim=0),
            requires_grad=False,
        )
        self.output_bias = nn.Parameter(proj_bias, requires_grad=False)

    def _split_heads(self, tensor: torch.Tensor, width: int) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, width).permute(0, 2, 1, 3)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, width = tensor.shape
        return (
            tensor.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, num_heads * width)
        )

    def forward(
        self,
        hidden_states: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        del past_key_values, cache_position, encoder_hidden_states, encoder_attention_mask, kwargs
        if hidden_states is None:
            raise ValueError("hidden_states must not be None")

        query = hidden_states @ self.q_weight + self.q_bias
        key = hidden_states @ self.k_weight + self.k_bias
        value = hidden_states @ self.low_rank_value_weight + self.low_rank_value_bias

        query = self._split_heads(query, self.head_dim)
        key = self._split_heads(key, self.head_dim)
        value = self._split_heads(value, self.rank)

        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_scores = attn_scores / (self.head_dim ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            attn_scores = attn_scores / float(self.layer_idx + 1)

        q_len, k_len = attn_scores.size(-2), attn_scores.size(-1)
        causal_mask = torch.tril(
            torch.ones((q_len, k_len), device=attn_scores.device, dtype=torch.bool)
        )
        attn_scores = attn_scores.masked_fill(
            ~causal_mask,
            torch.finfo(attn_scores.dtype).min,
        )
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_scores = attn_scores + attention_mask[:, None, None, :]
            else:
                attn_scores = attn_scores + attention_mask

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = attn_output @ self.low_rank_output_weight + self.output_bias
        attn_output = self.resid_dropout(attn_output)

        outputs: tuple[torch.Tensor, ...] = (attn_output, None)
        if output_attentions:
            outputs = (attn_output, attn_probs)
        return outputs


class GPT2Adapter:
    def __init__(self, model: GPT2LMHeadModel, tokenizer: SimpleCharTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = model.config.n_embd
        self.num_layers = model.config.n_layer
        self.num_heads = model.config.n_head
        self.head_dim = self.hidden_size // self.num_heads

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPT2Adapter":
        pretrained_name = config.get("pretrained_name")
        if pretrained_name:
            raise NotImplementedError(
                "Pretrained loading is intentionally deferred in the minimal offline scaffold."
            )
        model = GPT2LMHeadModel(GPT2Config(**config["gpt2_config"]))
        tokenizer_cfg = config.get("tokenizer", {})
        tokenizer = SimpleCharTokenizer(vocab_size=int(tokenizer_cfg.get("vocab_size", 256)))
        return cls(model, tokenizer)

    def clone(self) -> "GPT2Adapter":
        return GPT2Adapter(deepcopy(self.model), deepcopy(self.tokenizer))

    def to(self, device: torch.device | str) -> "GPT2Adapter":
        self.model.to(device)
        return self

    def get_attention_layers(self):
        return [block.attn for block in self.model.transformer.h]

    def extract_value_heads(self, c_attn_output: torch.Tensor) -> torch.Tensor:
        values = c_attn_output[..., 2 * self.hidden_size :]
        return values.reshape(-1, self.num_heads, self.head_dim)

    def extract_output_heads(self, c_proj_input: torch.Tensor) -> torch.Tensor:
        return c_proj_input.reshape(-1, self.num_heads, self.head_dim)

    def apply_compression(self, spec: CompressionSpec) -> "GPT2Adapter":
        clone = self.clone()
        clone.model.transformer.h[spec.layer_index].attn = CompressedGPT2Attention(
            clone.model.transformer.h[spec.layer_index].attn,
            spec.bases,
        )
        return clone
