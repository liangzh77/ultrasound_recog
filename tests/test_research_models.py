import torch
from torch import nn

from src.research_models import MaskedMeanClassifier, create_timm_encoder


class MeanChannelEncoder(nn.Module):
    num_features = 3

    def forward(self, images):
        return images.mean(dim=(2, 3))


def test_masked_mean_classifier_ignores_padding_and_averages_probabilities():
    model = MaskedMeanClassifier(
        MeanChannelEncoder(),
        feature_dim=3,
        num_classes=3,
        dropout=0.0,
    )
    with torch.no_grad():
        model.head.weight.copy_(torch.eye(3))
        model.head.bias.zero_()
    images = torch.zeros((2, 2, 3, 2, 2))
    images[0, 0, 0] = 3.0
    images[0, 1, 1] = 3.0
    images[1, 0, 2] = 3.0
    images[1, 1, 0] = 100.0
    mask = torch.tensor([[True, True], [True, False]])

    output = model(images, mask)

    first_expected = (
        torch.softmax(torch.tensor([3.0, 0.0, 0.0]), dim=0)
        + torch.softmax(torch.tensor([0.0, 3.0, 0.0]), dim=0)
    ) / 2
    second_expected = torch.softmax(torch.tensor([0.0, 0.0, 3.0]), dim=0)
    assert torch.allclose(output["patient_probabilities"][0], first_expected)
    assert torch.allclose(output["patient_probabilities"][1], second_expected)
    assert torch.allclose(output["patient_probabilities"].sum(dim=1), torch.ones(2))


def test_timm_encoder_exposes_pooled_feature_dimension_without_pretraining():
    encoder, feature_dim = create_timm_encoder(
        "efficientnet_b2.ra_in1k",
        pretrained=False,
    )

    with torch.inference_mode():
        features = encoder(torch.zeros((1, 3, 64, 64)))

    assert feature_dim == encoder.num_features
    assert features.shape == (1, feature_dim)


def test_masked_mean_classifier_chunked_inference_matches_single_pass():
    model = MaskedMeanClassifier(
        MeanChannelEncoder(),
        feature_dim=3,
        num_classes=3,
        dropout=0.0,
    )
    images = torch.rand((2, 5, 3, 4, 4))
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )

    with torch.inference_mode():
        single = model(images, mask)["patient_probabilities"]
        chunked = model(images, mask, instance_chunk_size=2)["patient_probabilities"]

    assert torch.allclose(single, chunked)
