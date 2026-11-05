import unittest

import numpy as np
import pandas as pd

from price_only.data import (
    PRICE_FIELDS,
    fit_standardizer,
    ordered_windows,
    sequence_features,
    split_development_rows,
    tree_features,
)
from price_only.evaluation import evaluate_predictions
from price_only.train import validate_prediction_set


class PriceOnlyPipelineTests(unittest.TestCase):
    def test_window_order_is_oldest_to_newest(self):
        row = {}
        for lag in range(1, 31):
            for offset, field in enumerate(PRICE_FIELDS):
                row[f"{field}_lag_{lag}"] = float(100 - lag + offset)
        window = ordered_windows(pd.DataFrame([row]))
        self.assertEqual(window.shape, (1, 30, 5))
        self.assertEqual(window[0, 0, 0], 70.0)  # lag_30
        self.assertEqual(window[0, -1, 0], 99.0)  # lag_1

    def test_split_excludes_outcomes_that_cross_years(self):
        frame = pd.DataFrame({
            "prediction_date": pd.to_datetime(
                ["2020-12-23", "2020-12-30", "2021-01-04", "2021-12-30"]
            ),
            "exit_date": pd.to_datetime(
                ["2020-12-31", "2021-01-07", "2021-01-11", "2022-01-07"]
            ),
        })
        train, validation = split_development_rows(frame, 2020, 2021)
        self.assertEqual(train.tolist(), [True, False, False, False])
        self.assertEqual(validation.tolist(), [False, False, True, False])

    def test_features_only_use_supplied_window_and_train_standardizer(self):
        windows = np.ones((2, 30, 5), dtype=np.float32)
        windows[:, :, :4] *= 100
        windows[:, :, 4] *= 1_000
        windows[1, :, :4] *= 2
        windows[1, :, 4] *= 10
        tree, names = tree_features(windows)
        self.assertEqual(tree.shape[0], 2)
        self.assertEqual(tree.shape[1], len(names))
        self.assertTrue(np.isfinite(tree).all())
        train_sequences = sequence_features(windows[:1], include_volume=True)
        validation_sequences = sequence_features(windows[1:], include_volume=True)
        mean, std = fit_standardizer(train_sequences)
        self.assertEqual(mean.shape, (1, 1, 5))
        self.assertGreater(float(((validation_sequences - mean) / std)[0, 0, 4]), 1.0)

    def test_daily_rank_scores_and_constant_baseline(self):
        keys = pd.DataFrame({
            "prediction_date": pd.to_datetime(
                ["2021-01-04"] * 3 + ["2021-01-05"] * 3
            )
        })
        actual = np.array([-0.02, 0.01, 0.04, 0.03, -0.01, 0.02])
        correct = evaluate_predictions(keys, actual, actual)
        reverse = evaluate_predictions(keys, actual, -actual)
        zero = evaluate_predictions(keys, actual, np.zeros_like(actual))
        self.assertAlmostEqual(correct["mean_daily_spearman"], 1.0)
        self.assertAlmostEqual(reverse["mean_daily_spearman"], -1.0)
        self.assertAlmostEqual(correct["mae_percentage_points"], 0.0)
        self.assertEqual(zero["mean_daily_spearman"], None)
        self.assertEqual(zero["ranked_days"], 0)
        self.assertEqual(correct["coverage"], 1.0)

    def test_prediction_keys_and_values_are_checked(self):
        keys = pd.DataFrame({
            "ticker": ["AMD", "WFC"],
            "prediction_date": pd.to_datetime(["2021-01-04", "2021-01-04"]),
        })
        validate_prediction_set(keys, {"model": np.array([0.01, -0.01])})
        with self.assertRaises(ValueError):
            validate_prediction_set(keys, {"model": np.array([0.01])})
        with self.assertRaises(ValueError):
            validate_prediction_set(keys, {"model": np.array([0.01, np.nan])})


if __name__ == "__main__":
    unittest.main()
