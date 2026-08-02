import json
from pathlib import Path

import yaml

from src.label_mapping import get_disease_from_label


ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY = ROOT / "configs" / "research" / "annotation_ontology_review_v0.yaml"
AUDIT_CONFIG = ROOT / "configs" / "research" / "annotation_supervision_audit.yaml"
MAPPING = (
    ROOT
    / "workspace"
    / "data"
    / "shared_derived"
    / "exp_2026-07_patient_multimodal_v1"
    / "annotations_62ecb01c4d77"
    / "category_mapping.json"
)


def test_review_ontology_covers_the_frozen_28_categories_without_disease_prefixes():
    ontology = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    audit = yaml.safe_load(AUDIT_CONFIG.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    categories = set(ontology["categories"])
    assert ontology["status"] == "clinical_review_required"
    assert categories == set(audit["category_roles"])
    assert categories == set(mapping["categories"])
    assert len(categories) == 28
    assert all(get_disease_from_label(category) is None for category in categories)


def test_review_ontology_does_not_treat_legacy_absence_as_a_negative_label():
    ontology = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    rules = ontology["global_rules"]

    assert rules["unlabeled_means_negative"] is False
    assert rules["negative_state_required"] == "absent_visible"
    assert "legacy_unknown" in rules["nonnegative_states"]
    assert rules["overlapping_instances_allowed"] is True


def test_every_category_has_a_supported_review_action():
    ontology = yaml.safe_load(ONTOLOGY.read_text(encoding="utf-8"))
    families = set(ontology["candidate_families"])
    permitted_tiers = {
        "robust_multifold",
        "limited_multifold",
        "insufficient_multifold",
    }

    for category, contract in ontology["categories"].items():
        assert contract["family"] in families, category
        assert contract["support_tier"] in permitted_tiers, category
        assert int(contract["patients"]) > 0, category
        assert contract["action"], category
