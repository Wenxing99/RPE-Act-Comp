from __future__ import annotations

import math

import torch


def evaluate_causal_lm(
    adapter,
    texts: list[str],
    *,
    max_length: int,
    device: str,
    teacher_adapter=None,
) -> dict[str, float]:
    tokenized = adapter.tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    labels = tokenized["input_ids"].clone()

    adapter.model.eval()
    with torch.no_grad():
        outputs = adapter.model(**tokenized, labels=labels)
        logits = outputs.logits
        nll = float(outputs.loss.item())

    metrics = {
        "nll": nll,
        "perplexity": float(math.exp(min(20.0, nll))),
        "num_tokens": int(tokenized["attention_mask"].sum().item()),
    }

    if teacher_adapter is not None:
        teacher_adapter.model.eval()
        with torch.no_grad():
            teacher_logits = teacher_adapter.model(**tokenized).logits
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        student_log_probs = torch.log_softmax(logits, dim=-1)
        kl = (
            teacher_probs
            * (torch.log(teacher_probs.clamp_min(1e-9)) - student_log_probs)
        ).sum(dim=-1)
        metrics["teacher_logit_kl"] = float(kl.mean().item())

    return metrics
