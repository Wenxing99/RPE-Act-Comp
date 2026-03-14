from transformers import GPT2Config, GPT2LMHeadModel

from data.text_data import SimpleCharTokenizer
from hooks.vo_hooks import GPT2VOActivationCollector
from models.gpt2_adapter import GPT2Adapter


def test_hook_shapes_match_per_head_layout():
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=128,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_positions=32,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )
    adapter = GPT2Adapter(model, SimpleCharTokenizer(vocab_size=128))
    collector = GPT2VOActivationCollector(adapter)
    collector.register()
    tokenized = adapter.tokenizer(
        ["hook shape test"],
        return_tensors="pt",
        truncation=True,
        max_length=16,
    )
    model(**tokenized)
    collector.remove()

    stacked = collector.stacked()
    value = stacked["value"][0]
    output = stacked["output"][0]
    assert value.shape[-2:] == (adapter.num_heads, adapter.head_dim)
    assert output.shape[-2:] == (adapter.num_heads, adapter.head_dim)
