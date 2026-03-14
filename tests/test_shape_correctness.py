from experiments.collect_activations import collect_head_activations
from models.gpt2_adapter import GPT2Adapter
from utils.config import load_yaml


def test_activation_bundle_has_expected_metadata_and_shapes():
    model_config = load_yaml("configs/model/tiny_random_gpt2.yaml")
    adapter = GPT2Adapter.from_config(model_config).to("cpu")
    bundle = collect_head_activations(
        adapter,
        ["shape correctness test", "second sample"],
        max_length=16,
        device="cpu",
    )

    assert bundle["metadata"]["num_layers"] == 2
    assert bundle["metadata"]["num_heads"] == 4
    assert bundle["metadata"]["head_dim"] == 16
    assert bundle["activations"]["value"][0].shape[-2:] == (4, 16)
    assert bundle["activations"]["output"][0].shape[-2:] == (4, 16)
