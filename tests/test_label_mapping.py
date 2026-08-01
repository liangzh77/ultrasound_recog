import unittest

from src.label_mapping import fix_label


class DiseaseIndependentRegionLabelTests(unittest.TestCase):
    def test_removes_disease_prefix_from_same_anatomy(self):
        labels = [
            "N骨皮质",
            "OA骨皮质",
            "RA骨皮质",
            "GA骨皮质",
            "SPA-骨皮质",
        ]

        self.assertEqual(
            [fix_label(label) for label in labels],
            ["骨皮质"] * len(labels),
        )

    def test_removes_non_abbreviation_disease_prefixes(self):
        self.assertEqual(fix_label("损伤-积液"), "积液")
        self.assertEqual(fix_label("滑膜囊肿-腘窝囊肿"), "腘窝囊肿")

    def test_fixes_reintroduced_spa_typo_before_removing_prefix(self):
        self.assertEqual(
            fix_label("SPA-斌下肾囊炎", "脊柱关节炎"),
            "髌下深囊炎",
        )

    def test_fixes_new_ga_tendon_typo_before_removing_prefix(self):
        self.assertEqual(
            fix_label("GA股二头肌建", "痛风性关节炎"),
            "股二头肌腱",
        )

    def test_fixes_unprefixed_typo_without_inventing_a_disease_prefix(self):
        self.assertEqual(
            fix_label("内测半月板", "正常"),
            "内侧半月板",
        )

    def test_preserves_specific_unprefixed_region_meaning(self):
        self.assertEqual(fix_label("半月板囊肿", "损伤"), "半月板囊肿")
        self.assertEqual(fix_label("腘窝囊肿", "骨性关节炎"), "腘窝囊肿")

    def test_unifies_prepatellar_superficial_fascia_wording(self):
        self.assertEqual(fix_label("N髌前浅筋膜"), "髌前浅筋膜")
        self.assertEqual(fix_label("损伤-髌骨前浅筋膜"), "髌前浅筋膜")

    def test_normalization_is_idempotent(self):
        once = fix_label("RA骨赘", "类风湿性关节炎")
        self.assertEqual(fix_label(once, "类风湿性关节炎"), once)


if __name__ == "__main__":
    unittest.main()
