"""Compare additional signals and training cohorts on the 50-company project grid."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from price_only.evaluation import evaluate_predictions
from price_only_v2.build_dataset import sha256
from price_refinement.experiment import DATA, calibrate, fit_model, pack_predictions, predict_model, read_research
from price_refinement.study_data import ARCHIVE, EARLY, FROZEN, MEMBERS, MEMBERS_REV, membership, prepare_prices
from price_refinement.study_features import build_panels

OUT = Path("results/price_study")
MODEL = Path("models/price_study")
BASE = ["relative_1", "relative_4", "relative_9", "relative_20", "relative_29",
        "intraday_5", "overnight_5", "volatility_20", "range_5", "drawdown_20",
        "risk_scaled_relative_4", "spy_volatility_20", "spy_return_20"]
MARKET = [f"market_{w}_{n}" for w in (60, 120) for n in (1, 5, 20)]
SECTOR = [f"sector_{w}_{n}" for w in (60, 120) for n in (1, 5, 20)]
LONG = [f"trend_{n}" for n in (60, 120, 240)]
REGIME = ["market_volatility_relative", "regime_rel1", "regime_rel4", "regime_residual1", "regime_residual5"]
SPECS = (
    ("ridge_reference", "ridge", BASE, 2017, False),
    ("market_residual", "ridge", BASE + MARKET, 2017, False),
    ("market_sector_residual", "ridge", BASE + MARKET + SECTOR, 2017, False),
    ("volatility_interactions", "ridge", BASE + REGIME, 2017, False),
    ("residual_and_volatility", "ridge", BASE + MARKET + SECTOR + REGIME, 2017, False),
    ("longer_lookbacks", "ridge", BASE + LONG, 2017, False),
    ("longer_history", "ridge", BASE, 2010, False),
    ("longer_history_residual", "ridge", BASE + MARKET, 2010, False),
    ("broader_universe", "ridge", BASE, 2017, True),
    ("broader_and_earlier", "ridge", BASE, 2010, True),
    ("broader_residual_volatility", "ridge", BASE + MARKET + LONG + REGIME, 2010, True),
    ("broader_robust_residual", "huber", BASE + MARKET + LONG + REGIME, 2010, True),
    ("broader_rank_residual", "ridge_rank_target", BASE + MARKET + LONG + REGIME, 2010, True),
    ("broader_tree_residual", "tree", BASE + MARKET + LONG + REGIME, 2010, True),
    ("tail_classifier", "tail", BASE + MARKET + SECTOR + REGIME, 2017, False),
    ("broader_tail_classifier", "tail", BASE + MARKET + LONG + REGIME, 2010, True),
)
YEARS = (2018, 2019, 2020, 2021)


def add_interactions(frame):
    v = frame.market_volatility_relative
    return frame.assign(regime_rel1=v * frame.relative_1, regime_rel4=v * frame.relative_4,
                        regime_residual1=v * frame.market_60_1, regime_residual5=v * frame.market_60_5)


def day_scores(frame, predicted):
    metrics = []
    for day, group in frame.assign(pred=predicted).groupby("prediction_date"):
        p, y = group.pred.to_numpy(), group.target_excess_return.to_numpy()
        if np.ptp(p) == 0:
            rank, spread = np.nan, np.nan
        else:
            rank = np.corrcoef(rankdata(p), rankdata(y))[0, 1]
            # Average tied groups so ticker order cannot drive the top-five result.
            scores = pd.DataFrame({"score": p, "actual": y})
            scores["actual"] = scores.groupby("score").actual.transform("mean")
            sorted_y = scores.sort_values("score", ascending=False).actual.to_numpy()
            middle = (len(y) - 10) // 2
            spread = (sorted_y[:5].mean() - sorted_y[middle:middle + 10].mean()) * 100
        metrics.append((day, rank, spread))
    return pd.DataFrame(metrics, columns=["prediction_date", "rank", "top5_minus_middle10_pp"])


def interval(values, block_size=20, alpha=.05):
    """Resample consecutive trading sessions to account for overlapping outcomes."""
    values = np.asarray(values)
    if len(values) < block_size or not np.isfinite(values).any():
        return [None, None]
    rng = np.random.default_rng(20261112)
    indices = rng.integers(0, len(values)-block_size+1, size=(10000, int(np.ceil(len(values)/block_size))))
    draws = (indices[:, :, None] + np.arange(block_size)).reshape(len(indices), -1)[:, :len(values)]
    return np.nanquantile(np.nanmean(values[draws], axis=1), [alpha/2, 1-alpha/2]).tolist()


def train_panel(project, extras, valid, year, kind, columns, first_year, broader):
    start, cutoff = f"{first_year}-01-01", f"{year}-01-01"
    all_rows = pd.concat([project, extras], ignore_index=True) if broader else project
    train = all_rows.loc[(all_rows.prediction_date >= start) & (all_rows.exit_date < cutoff)]
    # Keep the original 30-session eligibility; missing long-history features are neutralized.
    train = train.sort_values(["prediction_date", "ticker"]).reset_index(drop=True)
    x = train[columns].fillna(0)
    if kind == "tail":
        dates = train.prediction_date
        cutoff = dates.drop_duplicates().iloc[-63]
        fit = train.exit_date < cutoff
        cal = dates >= cutoff
        rank = train.target_excess_return.groupby(dates).rank(pct=True)
        tail = (rank <= .2) | (rank > .8)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=200))
        estimator.fit(x.loc[fit & tail], rank[fit & tail] > .8)
        offset, scale = calibrate(estimator.decision_function(x.loc[cal]), train.target_excess_return[cal].to_numpy())
        model = {"kind": "tail", "model": estimator, "calibration": (offset, scale)}
    else:
        model = fit_model(kind, x, train.target_excess_return.to_numpy(), train, 42)
    pred = study_prediction(model, valid[columns].fillna(0))
    return pred, model, len(train), train.ticker.nunique()


def study_prediction(bundle, features):
    if bundle["kind"] == "tail":
        offset, scale = bundle["calibration"]
        return offset + scale * bundle["model"].decision_function(features)
    return predict_model(bundle, features)


def check_frozen(project, dataset):
    reference = read_research(dataset)
    check = reference.loc[reference.prediction_date.dt.year.isin(YEARS)
                          & (reference.prediction_date.dt.year == reference.exit_date.dt.year)]
    keyed = check.set_index(["prediction_date", "ticker"])
    obtained = project.set_index(["prediction_date", "ticker"])
    if not keyed.index.isin(obtained.index).all():
        raise ValueError("Missing frozen project validation rows")
    values = obtained.loc[keyed.index]
    gap = float(np.max(np.abs(values.target_excess_return - keyed.target_excess_return)))
    if gap > 1e-8:
        raise ValueError(f"Frozen target mismatch: {gap}")
    return check[["ticker", "prediction_date", "exit_date", "target_excess_return"]], gap


def run(dataset, output, model_dir):
    groups = pd.read_csv("CURRENT_COMPANIES.csv").set_index("ticker").sector
    prices, discrepancy = prepare_prices()
    member_dates = membership()
    project, extras, unavailable, reused = build_panels(prices, groups, member_dates)
    project, extras = add_interactions(project), add_interactions(extras)
    frozen, gap = check_frozen(project, dataset)
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    logs, daily_logs, predictions, saved = [], [], {}, {}
    for year in YEARS:
        valid = frozen.loc[frozen.prediction_date.dt.year == year, ["prediction_date", "ticker"]].merge(
            project, on=["prediction_date", "ticker"], how="left", validate="one_to_one")
        fixed = {"zero": np.zeros(len(valid)), "trend": .05 * valid.trend.to_numpy()}
        if year >= 2020:
            earlier = pd.DataFrame(logs)
            selected = (earlier[earlier.model_name.isin([s[0] for s in SPECS])]
                        .groupby("model_name").mean_daily_spearman.mean().idxmax())
            saved[year] = selected
        for name, kind, cols, start, broader in [("zero", None, None, None, False),
                                                 ("trend", None, None, None, False), *SPECS]:
            if kind is None:
                pred, model, nrows, ntickers = fixed[name], None, 0, 0
            else:
                pred, model, nrows, ntickers = train_panel(project, extras, valid, year, kind, cols, start, broader)
            if name == saved.get(year):
                predictions[year] = pack_predictions(valid, {"price_study_candidate": pred})
                joblib.dump({"model": model, "name": name, "columns": cols,
                             "last_training_exit": pd.Timestamp(f"{year - 1}-12-31")},
                            model_dir / f"candidate_{year}.joblib")
            result = evaluate_predictions(valid, valid.target_excess_return, pred)
            daily = day_scores(valid, pred)
            logs.append({"year": year, "model_name": name, "train_rows": nrows,
                         "train_tickers": ntickers, "top5_minus_middle10_pp": daily.top5_minus_middle10_pp.mean(), **result})
            daily_logs.append(daily.assign(year=year, model_name=name))
        print(f"Finished {year}: {len(valid):,} validation rows. Selection from earlier folds: {saved.get(year)}", flush=True)
    summary = pd.DataFrame(logs)
    daily = pd.concat(daily_logs, ignore_index=True)
    summary.to_csv(output / "experiment_log.csv", index=False)
    daily.to_csv(output / "daily_metrics.csv", index=False)
    pd.concat(predictions.values()).to_csv(output / "predictions_2020_2021.csv", index=False, float_format="%.12g")
    selected = saved[2021]
    v = daily[daily.year == 2021].pivot(index="prediction_date", columns="model_name", values="rank")
    intervals = []
    for name in ["trend", *[s[0] for s in SPECS]]:
        ranks = v[name].to_numpy()
        spread = daily.loc[(daily.year == 2021) & (daily.model_name == name), "top5_minus_middle10_pp"]
        rank_low, rank_high = interval(ranks)
        delta_low, delta_high = interval(v[name]-v.trend)
        spread_low, spread_high = interval(spread)
        intervals.append({"model_name": name, "rank_95_low": rank_low,
                          "rank_95_high": rank_high,
                          "delta_vs_trend": float((v[name]-v.trend).mean()),
                          "delta_95_low": delta_low, "delta_95_high": delta_high,
                          "top5_95_low_pp": spread_low, "top5_95_high_pp": spread_high})
    pd.DataFrame(intervals).to_csv(output / "intervals_2021.csv", index=False)
    counts = extras.ticker.value_counts()
    coverage = pd.DataFrame([{"ticker": name, "missing_archive_or_rows": name in unavailable,
                              "excluded_for_ticker_reuse": name in reused,
                              "project_ticker": name.replace(".", "-") in set(groups.index) or name == "FB",
                              "training_rows": int(counts.get(name, 0))} for name in sorted(member_dates)])
    coverage.to_csv(output / "membership_coverage.csv", index=False)
    manifest = {"inputs_sha256": {p.name: sha256(p) for p in (dataset, FROZEN, EARLY, MEMBERS, ARCHIVE)},
                "membership_revision": MEMBERS_REV, "max_join_adjustment_relative_spread": discrepancy,
                "frozen_target_max_delta": gap, "extra_training_rows": len(extras),
                "extra_tickers": int(extras.ticker.nunique()), "unavailable_membership_symbols": len(unavailable),
                "excluded_reused_ticker_strings": reused, "specifications": SPECS,
                "selected_on_earlier_folds": saved, "reserved_test_scored": False,
                "uncertainty": "10,000 bootstrap samples of adjacent 20-session blocks, seed 20261112"}
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.pivot(index="model_name", columns="year", values="mean_daily_spearman").to_string())
    print("2021 selection:", selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    args = parser.parse_args()
    run(args.data, args.output, args.model_dir)
