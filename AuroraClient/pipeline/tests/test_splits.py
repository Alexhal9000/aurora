from django.test import TestCase

from pipeline.aiFineTuning.splits import (
    SplitError,
    assert_no_subject_leakage,
    k_fold_splits,
    subject_level_split,
)


class SubjectSplitTests(TestCase):
    def test_reproducible_seed(self):
        subjects = [f"S{i}" for i in range(10)]
        first = subject_level_split(subjects, seed=7)
        second = subject_level_split(subjects, seed=7)
        self.assertEqual(first, second)

    def test_no_leakage_and_full_coverage(self):
        subjects = [f"S{i}" for i in range(10)]
        assignments = subject_level_split(subjects, seed=3)
        assert_no_subject_leakage(assignments)
        combined = assignments["train"] + assignments["validation"] + assignments["test"]
        self.assertEqual(sorted(combined), sorted(subjects))
        self.assertTrue(assignments["train"])
        self.assertTrue(assignments["validation"])
        self.assertTrue(assignments["test"])

    def test_different_seeds_change_assignment(self):
        subjects = [f"S{i}" for i in range(12)]
        a = subject_level_split(subjects, seed=1)
        b = subject_level_split(subjects, seed=2)
        self.assertNotEqual(a["train"], b["train"])

    def test_kfold_no_leakage(self):
        subjects = [f"S{i}" for i in range(8)]
        folds = k_fold_splits(subjects, k=4, seed=11)
        self.assertEqual(len(folds), 4)
        for fold in folds:
            assert_no_subject_leakage({
                "train": fold["train"],
                "validation": fold["validation"],
                "test": fold["test"],
            })
            self.assertEqual(sorted(fold["train"] + fold["validation"]), sorted(subjects))

    def test_rejects_too_few_subjects(self):
        with self.assertRaises(SplitError):
            subject_level_split(["only"], seed=1)
