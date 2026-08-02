from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.research_config import (
    assert_configs_differ_only,
    load_research_config,
    resolve_pretrained_weights,
)
from src.research_models import MaskedMeanClassifier
from src.research_mil import GatedAttentionMILClassifier
from src.research_training import (
    EarlyStopping,
    make_patient_balanced_sampler,
    previous_elapsed_hours,
    run_patient_epoch,
    warmup_cosine_multiplier,
)


ROOT = Path(__file__).resolve().parent.parent


class TinyEncoder(nn.Module):
    num_features = 3

    def forward(self, images):
        return images.mean(dim=(2, 3))


class TinyBags(Dataset):
    labels = (0, 0, 0, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        target = self.labels[index]
        images = torch.zeros((1, 3, 4, 4))
        images[:, target] = 1.0
        return {
            "images": images,
            "instance_mask": torch.ones((1,), dtype=torch.bool),
            "targets": torch.tensor(target),
            "person_keys": f"P{index}",
            "image_keys": [f"I{index}"],
        }


def _collate(rows):
    return {
        "images": torch.stack([row["images"] for row in rows]),
        "instance_mask": torch.stack([row["instance_mask"] for row in rows]),
        "targets": torch.stack([row["targets"] for row in rows]),
        "person_keys": [row["person_keys"] for row in rows],
        "image_keys": [row["image_keys"] for row in rows],
    }


def test_e0_e1_configs_differ_only_by_experiment_and_input_mode():
    e0 = load_research_config(ROOT / "configs/research/e0_full_mean_b2.yaml")
    e1 = load_research_config(ROOT / "configs/research/e1_roi_mean_b2.yaml")

    assert e0["input_mode"] == "full"
    assert e1["input_mode"] == "roi"
    assert_configs_differ_only(e0, e1, {"experiment_code", "input_mode"})


def test_e1s_changes_only_the_preregistered_resize_mode():
    e1 = load_research_config(ROOT / "configs/research/e1_roi_mean_b2.yaml")
    e1s = load_research_config(ROOT / "configs/research/e1s_roi_stretch_mean_b2.yaml")

    comparable_e1 = deepcopy(e1)
    comparable_e1s = deepcopy(e1s)
    comparable_e1.pop("experiment_code")
    comparable_e1s.pop("experiment_code")
    comparable_e1["data"]["resize_mode"] = "stretch"

    assert e1["data"]["resize_mode"] == "letterbox"
    assert e1s["data"]["resize_mode"] == "stretch"
    assert comparable_e1 == comparable_e1s


def test_e2_changes_only_patient_aggregation_from_e1():
    e1 = load_research_config(ROOT / "configs/research/e1_roi_mean_b2.yaml")
    e2 = load_research_config(
        ROOT / "configs/research/e2_roi_gated_attention_b2.yaml"
    )

    comparable_e1 = deepcopy(e1)
    comparable_e2 = deepcopy(e2)
    comparable_e1.pop("experiment_code")
    comparable_e2.pop("experiment_code")
    comparable_e2["model"].pop("aggregation")
    comparable_e2["model"].pop("attention_dim")
    comparable_e2["model"].pop("attention_collapse_threshold")
    comparable_e2["model"].pop("max_multi_image_collapse_rate")

    assert e2["model"]["aggregation"] == "gated_attention"
    assert comparable_e1 == comparable_e2


def test_e2r_changes_only_preregistered_attention_regularization_from_e2():
    e2 = load_research_config(
        ROOT / "configs/research/e2_roi_gated_attention_b2.yaml"
    )
    e2r = load_research_config(
        ROOT / "configs/research/e2r_roi_gated_attention_entropy_b2.yaml"
    )
    comparable_e2 = deepcopy(e2)
    comparable_e2r = deepcopy(e2r)
    comparable_e2.pop("experiment_code")
    comparable_e2r.pop("experiment_code")
    comparable_e2["training"]["attention_kl_weight"] = 0.05

    assert e2r["training"]["attention_kl_weight"] == 0.05
    assert comparable_e2 == comparable_e2r


def test_pretrained_weight_path_is_local_and_hash_verified(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"known weights")
    config = {
        "model": {
            "pretrained_path": "weights.bin",
            "pretrained_sha256": (
                "752bf40592a4d8d9399e2342ae9534157e986085711d0af1f49f0b62879d57dd"
            ),
        }
    }

    assert resolve_pretrained_weights(config, tmp_path) == weights.resolve()

    config["model"]["pretrained_sha256"] = "0" * 64
    try:
        resolve_pretrained_weights(config, tmp_path)
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("A mismatched pretrained hash must be rejected")


def test_patient_balanced_sampler_uses_inverse_patient_class_frequency():
    sampler = make_patient_balanced_sampler(TinyBags(), seed=7)

    assert sampler.num_samples == 4
    assert sampler.replacement is True
    assert sampler.weights.tolist() == [1 / 3, 1 / 3, 1 / 3, 1.0]


def test_training_and_validation_epoch_return_patient_level_outputs():
    model = MaskedMeanClassifier(TinyEncoder(), 3, 2, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loader = DataLoader(TinyBags(), batch_size=2, collate_fn=_collate)

    trained = run_patient_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        accumulation_steps=2,
        amp=False,
    )
    validated = run_patient_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        amp=False,
    )

    assert trained["prediction_level"] == "patient"
    assert validated["prediction_level"] == "patient"
    assert validated["probabilities"].shape == (4, 2)
    assert validated["targets"].tolist() == [0, 0, 0, 1]
    assert validated["person_keys"] == ["P0", "P1", "P2", "P3"]
    assert torch.allclose(validated["probabilities"].sum(dim=1), torch.ones(4))


def test_patient_epoch_collects_mil_attention_without_second_pass():
    model = GatedAttentionMILClassifier(
        TinyEncoder(),
        feature_dim=3,
        num_classes=2,
        attention_dim=4,
        dropout=0.0,
    )
    loader = DataLoader(TinyBags(), batch_size=2, collate_fn=_collate)

    result = run_patient_epoch(
        model,
        loader,
        device=torch.device("cpu"),
        optimizer=None,
        amp=False,
        collect_attention=True,
    )

    assert len(result["attention_summaries"]) == 4
    assert result["attention_summaries"][0]["person_key"] == "P0"
    assert result["attention_summaries"][0]["image_keys"] == ["I0"]
    assert result["attention_summaries"][0]["attention_weights"] == [1.0]


def test_early_stopping_tracks_best_epoch_and_patience():
    stopping = EarlyStopping(patience=2)

    assert stopping.update(epoch=0, score=0.4) == (True, False)
    assert stopping.update(epoch=1, score=0.3) == (False, False)
    assert stopping.update(epoch=2, score=0.3) == (False, True)
    assert stopping.best_epoch == 0
    assert stopping.best_score == 0.4


def test_resume_timing_uses_accumulated_total_and_supports_legacy_history():
    history = [
        {"elapsed_hours": 0.1},
        {"elapsed_hours": 0.2},
        {"elapsed_hours": 0.05, "elapsed_hours_total": 0.25},
    ]

    assert previous_elapsed_hours(history) == 0.25
    assert previous_elapsed_hours([]) == 0.0


def test_short_pilot_scheduler_does_not_zero_lr_before_final_epoch():
    values = [warmup_cosine_multiplier(epoch, 5, 3) for epoch in range(5)]

    assert values[:4] == [1 / 3, 2 / 3, 1.0, 1.0]
    assert values[4] == 0.5
