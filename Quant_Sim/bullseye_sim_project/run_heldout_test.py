"""Run the held-out test ONCE with the frozen config. Refuses edited configs and repeat runs."""
import argparse, os
from bullseye_sim import verify_frozen, run_simulation
from bullseye_sim.io import load_prices, load_predictions, load_period_benchmarks

ap = argparse.ArgumentParser()
ap.add_argument("--prices", required=True)
ap.add_argument("--preds", required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--benchmarks", default=None)
ap.add_argument("--frozen", default="outputs/validation/final_config.yaml")
ap.add_argument("--out-dir", default="outputs/test")
a = ap.parse_args()

cfg = verify_frozen(a.frozen)
marker = os.path.join(a.out_dir, ".test_run_done")
if os.path.exists(marker):
    raise SystemExit("Test already run for this output folder. The held-out test is calculated once.")
res = run_simulation(load_predictions(a.preds, a.model), load_prices(a.prices), cfg, cfg.test_start, cfg.test_end,
                     period_benchmarks=load_period_benchmarks(a.benchmarks) if a.benchmarks else None)
os.makedirs(a.out_dir, exist_ok=True)
res.trades.to_csv(f"{a.out_dir}/trades.csv", index=False)
res.daily.to_csv(f"{a.out_dir}/daily.csv", index=False)
res.holdings.to_csv(f"{a.out_dir}/holdings.csv", index=False)
res.log.to_csv(f"{a.out_dir}/skipped_log.csv", index=False)
res.periods.to_csv(f"{a.out_dir}/periods.csv", index=False)
import json
json.dump({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in res.metrics.items()},
          open(f"{a.out_dir}/metrics.json", "w"), indent=2)
open(marker, "w").write("done")
print(json.dumps(res.metrics, indent=2, default=float))
