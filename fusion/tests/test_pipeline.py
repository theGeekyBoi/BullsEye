import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fusion.combine import combine, export_predictions, read_predictions
from fusion.evaluate import evaluate, read_targets, top_vs_middle


class FusionTests(unittest.TestCase):
    def test_price_fallback_keeps_full_grid_and_gates_unusable_news(self):
        tickers = ["A", "B", "C", "D"]
        prices = pd.DataFrame({"ticker": tickers, "prediction_date": "2021-02-01",
                               "predicted_excess_return": [.02, -.01, .03, .04]})
        news = prices.iloc[[0, 2, 3]].assign(predicted_excess_return=[-.04, .90, .95])
        coverage = prices.iloc[:3][["ticker", "prediction_date"]].assign(news_eligible=[True, True, False])
        targets = prices[["ticker", "prediction_date"]].assign(target_excess_return=[.03, -.02, .01, .04])

        result = combine(prices, prices, news, coverage)
        self.assertEqual(result.news_status.tolist(), [
            "blended", "missing_news_forecast", "ineligible", "missing_coverage_row",
        ])
        self.assertEqual(export_predictions(result).predicted_excess_return.tolist(), [-.01, -.01, .03, .04])
        report, counts = evaluate(result, targets, same_as_trend=True)
        self.assertEqual(set(report.loc[report.scope == "matched_news", "rows"]), {1})
        self.assertEqual(set(report.loc[report.scope == "full_price_grid", "rows"]), {4})
        self.assertAlmostEqual(counts.fallback_rate.iloc[0], .75)

    def test_duplicate_or_malformed_forecasts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            header = "ticker,prediction_date,model_name,predicted_excess_return\n"
            row = "A,2021-01-04,news,0.02\n"
            path.write_text(header + row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate ticker/date"):
                read_predictions(path, "news", {"A"})
            path.write_text(header + "A,2021-1-4,news,0.02\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
                read_predictions(path, "news", {"A"})

    def test_development_loader_excludes_future_and_boundary_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.csv"
            path.write_text(
                "ticker,prediction_date,exit_date,target_excess_return\n"
                "A,2021-12-23,2021-12-31,0.05\n"
                "A,2021-12-30,2022-01-06,excluded\n"
                "A,2022-01-03,2022-01-10,excluded\n", encoding="utf-8",
            )
            targets, count = read_targets(path, {"A"})
            self.assertEqual(targets.target_excess_return.tolist(), [.05])
            self.assertEqual(count, 1)

    def test_spread_does_not_overlap_and_averages_ties(self):
        frame = pd.DataFrame({"prediction_date": "2021-02-01",
                              "target_excess_return": [i / 100 for i in range(20)],
                              "prediction": [0] * 10 + [1] * 10})
        self.assertEqual(top_vs_middle(frame.iloc[1:], "prediction"), (None, 0))
        spread, days = top_vs_middle(frame, "prediction")
        self.assertAlmostEqual(spread, 5.0)
        self.assertEqual(days, 1)


if __name__ == "__main__":
    unittest.main()
