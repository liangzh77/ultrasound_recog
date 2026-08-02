import pytest
import torch
from torch import nn

from src.research_mil import (
    GatedAttentionMILClassifier,
    normalized_attention_kl_to_uniform,
    summarize_attention,
)


class MeanChannelEncoder(nn.Module):
    def forward(self, images):
        return images.mean(dim=(2, 3))


def _model():
    model = GatedAttentionMILClassifier(
        MeanChannelEncoder(),
        feature_dim=3,
        num_classes=2,
        attention_dim=4,
        dropout=0.0,
    )
    model.eval()
    return model


def test_gated_attention_is_bag_order_invariant_and_ignores_padding():
    model = _model()
    images = torch.rand((2, 4, 3, 3, 3))
    mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    images[0, 3] = 1000.0
    images[1, 2:] = 1000.0
    permutation = torch.tensor([2, 0, 3, 1])

    with torch.inference_mode():
        original = model(images, mask)
        permuted = model(images[:, permutation], mask[:, permutation])

    assert torch.allclose(
        original["patient_probabilities"],
        permuted["patient_probabilities"],
        atol=1e-6,
    )
    assert torch.allclose(
        original["attention_weights"].sum(dim=1), torch.ones(2)
    )
    assert torch.equal(original["attention_weights"][~mask], torch.zeros(3))


def test_gated_attention_supports_single_image_and_chunked_encoding():
    model = _model()
    images = torch.rand((2, 3, 3, 4, 4))
    mask = torch.tensor([[True, False, False], [True, True, True]])

    with torch.inference_mode():
        single_pass = model(images, mask)
        chunked = model(images, mask, instance_chunk_size=2)

    assert single_pass["attention_weights"][0].tolist() == [1.0, 0.0, 0.0]
    assert torch.allclose(
        single_pass["patient_probabilities"],
        chunked["patient_probabilities"],
        atol=1e-6,
    )


def test_attention_summary_excludes_single_image_bags_from_collapse_rate():
    result = summarize_attention(
        [
            {"attention_weights": [1.0]},
            {"attention_weights": [0.96, 0.04]},
            {"attention_weights": [0.6, 0.4]},
        ],
        collapse_threshold=0.95,
    )

    assert result["patients"] == 3
    assert result["single_image_patients"] == 1
    assert result["multi_image_patients"] == 2
    assert result["multi_image_collapse_rate"] == 0.5


def test_normalized_attention_kl_is_zero_for_uniform_and_single_bags():
    weights = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])

    assert normalized_attention_kl_to_uniform(weights, mask).item() == 0.0


def test_normalized_attention_kl_approaches_one_for_concentrated_bag():
    weights = torch.tensor([[0.999999, 0.000001]])
    mask = torch.tensor([[True, True]])

    penalty = normalized_attention_kl_to_uniform(weights, mask)

    assert 0.9999 < penalty.item() <= 1.0


def test_attention_summary_rejects_non_normalized_weights():
    with pytest.raises(ValueError, match="sum to one"):
        summarize_attention([{"attention_weights": [0.2, 0.2]}])
