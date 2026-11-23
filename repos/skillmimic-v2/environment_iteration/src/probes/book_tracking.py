"""Compare pickup tracking on saved place_book evaluation traces (CPU only).

    .venv/bin/python environment_iteration/src/probes/book_tracking.py fix-5 fix-6

t is the zero-based stored control-step index. The signed horizontal components
use the reference book's net XY direction in the pickup window. Position and
velocity summaries use live samples only. The reconstructed position factor
uses the stored trace positions, which can differ from the live reward inputs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import iterlib as L
from metrics import TASK_EPS_M  # noqa: E402


def quantiles(values):
    if not values.size:
        return "n/a"
    return " / ".join(f"{v:.3f}" for v in np.percentile(values, [10, 50, 90]))


def report(name, cell, epoch, weights, seeds):
    folder = Path(L.paths(name)["eval_dir"]) / cell
    keys = ("obj", "ref_obj", "alive")
    by_seed = {key: [] for key in keys}
    for seed in seeds:
        path = folder / f"e{epoch}_s{seed}.npz"
        if not path.is_file():
            raise SystemExit(f"Missing evaluation trace: {L.rel(path)}")
        with np.load(path) as trace:
            for key in keys:
                by_seed[key].append(trace[key])

    obj, ref, alive = (np.concatenate(by_seed[key], axis=1) for key in keys)
    with (Path(L.REPO) / L.load(name)["cfg_env"]).open() as config:
        env = yaml.safe_load(config)["env"]
    meta = L.read_json(folder / f"e{epoch}_s{seeds[0]}.json")["meta"]
    n_clip = meta["n_clip"]
    if not 86 <= n_clip <= len(obj):
        raise SystemExit(f"{name} {cell} has no complete t45–85 clip window")

    delta = obj - ref
    error = np.linalg.norm(delta, axis=-1)
    pickup = slice(45, 86)
    live = alive[pickup]
    in_window = live.any(axis=0)
    heading = ref[85, 0, :2] - ref[45, 0, :2]
    heading /= np.linalg.norm(heading)
    lateral = np.array([-heading[1], heading[0]])
    xy = delta[pickup, :, :2]
    dz = delta[pickup, :, 2]
    vertical_speed_error = np.diff(delta[44:86, :, 2], axis=0) * env["dataFPS"]
    speed_live = alive[44:85] & live
    crossing = (error[:n_clip] > TASK_EPS_M) & alive[:n_clip]
    crossed = crossing.any(axis=0)
    survived = alive[n_clip - 1]
    peak_error = np.where(live, error[pickup], -np.inf).max(axis=0)[in_window]
    peak_vertical = np.where(live, dz, -np.inf).max(axis=0)[in_window]
    print(f"{name} epoch {epoch} {cell}, n={alive.shape[1]}, seeds={seeds}")
    print("  t45–85 (p10 / p50 / p90)")
    print(f"    max 3D error per episode (m)  {quantiles(peak_error)} ({in_window.sum()} episodes)")
    print(f"    peak vertical lead (m)       {quantiles(peak_vertical)}")
    print(f"    signed along / lateral (m)    {quantiles((xy @ heading)[live])} | {quantiles((xy @ lateral)[live])}")
    print(f"    vertical speed error (m/s)   {quantiles(vertical_speed_error[speed_live])}")
    print(f"  first >10 cm while alive (t)   {quantiles(crossing.argmax(axis=0)[crossed])} ({crossed.sum()} episodes)")
    print(f"  final 3D error (m)             {quantiles(error[n_clip - 1, survived])} ({survived.sum()} survivors)")
    lift = slice(55, 76)
    lift_error = error[lift][alive[lift]]
    for weight in weights or [env["rewardWeights"]["op"]]:
        factor = np.exp(-weight * lift_error**2 / 3)
        near_zero = f"{(factor < 0.05).mean():.1%} below 0.05" if factor.size else "no live samples"
        print(f"  t55–75 position factor op={weight:g}  {quantiles(factor)} ({near_zero})")


def main():
    protocol = L.protocol()
    cells = {cell["cell"]: cell["seeds"] for cell in protocol["eval"]["final"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("iterations", nargs="+")
    parser.add_argument("--epoch", type=int, default=protocol["train"]["epochs"])
    parser.add_argument("--cell", choices=cells, default="stoch")
    parser.add_argument("--weights", type=float, nargs="+")
    args = parser.parse_args()
    for name in args.iterations:
        report(name, args.cell, args.epoch, args.weights, cells[args.cell])


if __name__ == "__main__":
    main()
