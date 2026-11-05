import unittest

import numpy as np
import pandas as pd

from price_only_v2.evaluate import weekly_portfolio
from price_only_v2.features import feature_matrix, sequence_matrix, split_mask


class PipelineTests(unittest.TestCase):
    def test_short_window_ignores_older_prices(self):
        company = np.full((2, 30, 5), 10.0, dtype=np.float32)
        market = np.full((2, 30, 5), 20.0, dtype=np.float32)
        company[:, :, 4] = 100
        market[:, :, 4] = 200
        company[1, :25, :4] *= 2
        sector = pd.Series(["Technology", "Technology"])
        short, _ = feature_matrix(company, market, sector, 5)
        full, _ = feature_matrix(company, market, sector, 30)
        np.testing.assert_allclose(short[0], short[1])
        self.assertFalse(np.allclose(full[0], full[1]))
        singleton, _ = feature_matrix(company[:1], market[:1], sector[:1], 5)
        self.assertEqual(short.shape[1], singleton.shape[1])

    def test_sequence_has_company_and_spy_in_time_order(self):
        company = np.ones((1, 30, 5), dtype=np.float32)
        market = np.ones((1, 30, 5), dtype=np.float32)*2
        company[:, :, 4] = 10
        market[:, :, 4] = 20
        company[0, 0, 3] = 0.5
        company[0, -1, 3] = 1.5
        self.assertEqual(sequence_matrix(company, market, False).shape, (1, 30, 8))
        self.assertEqual(sequence_matrix(company, market, True).shape, (1, 30, 10))
        self.assertLess(sequence_matrix(company, market, False)[0, 0, 3], 0)

    def test_boundary_outcomes_are_excluded(self):
        frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2020-12-23", "2020-12-30", "2021-01-04", "2021-12-29"]),
                              "exit_date": pd.to_datetime(["2020-12-31", "2021-01-07", "2021-01-11", "2022-01-06"])})
        train, valid = split_mask(frame, 2020, 2020, 2021)
        np.testing.assert_array_equal(train, [True, False, False, False])
        np.testing.assert_array_equal(valid, [False, False, True, False])

    def test_weekly_rule_only_buys_positive_forecasts(self):
        rows = pd.DataFrame({"prediction_date": pd.to_datetime(["2021-01-04"]*3),
                             "ticker": ["A", "B", "C"],
                             "predicted_excess_return": [-0.01, 0.0, 0.02],
                             "actual_company_return": [-0.5, -0.5, 0.03],
                             "actual_spy_return": [0.01]*3})
        result = weekly_portfolio(rows, 10)
        self.assertEqual(result["stock_positions_bought"], 1)
        self.assertAlmostEqual(result["model_compounded_return"], (5/15)*(.03-.002))


if __name__ == "__main__":
    unittest.main()
