"""metrics.episode_metrics must reproduce experiments/tools/evaluate.py exactly (CPU only, ~10 s).

    python environment_iteration/src/tests/test_metrics.py [--max 40]

Recomputes every stored evaluate.py episode (.npz trace + .json result) under experiments/results_paper_*/eval and
compares all metric fields. obj_z0 is not stored by evaluate.py; the book is at rest for the first ~50 frames, so
the median reference height at frame 0 equals the reset height (median: a noisy-start env, see
protocol.json eval.state_init_random_prob, has a different reference frame).
Also checks failure_events() runs on every episode and returns a known mode.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import iterlib as L  # noqa: E402
import metrics as MX  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40, help="max result files to check")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(L.REPO, "experiments/results_paper_*/eval/*/*/*.json")))[:a.max]
    if not files:
        print("[test_metrics] SKIP: no stored evaluate.py results found")
        return 0
    n_ep = n_bad = 0
    for jf in files:
        nf = jf[:-5] + ".npz"
        if not os.path.isfile(nf):
            continue
        r = json.load(open(jf))
        S = dict(np.load(nf))
        m = r["meta"]
        c = dict(n_clip=m["clip_frames"] - m["start_frame"], obj_z0=float(np.median(S["ref_obj"][0, :, 2])),
                 ref_lift=m["ref_lift"], thr=0.2, paper_z=0.9, paper_need=60)
        fall_t = np.array([e["fall_frame"] for e in r["episodes"]])
        ph = MX.ref_phases(MX.clean_ref(S, c["n_clip"]), c["obj_z0"])
        for e, want in enumerate(r["episodes"]):
            got = MX.episode_metrics(S, fall_t, c, e)
            n_ep += 1
            diff = {k: (want[k], got[k]) for k in want
                    if not (want[k] == got[k] or (isinstance(want[k], float) and abs(want[k] - got[k]) < 1e-9))}
            if diff:
                n_bad += 1
                if n_bad <= 5:
                    print(f"[test_metrics] MISMATCH {L.rel(jf)} env {e}: {diff}")
            fe = MX.failure_events(S, fall_t, c, e, ph)
            assert fe["mode"] in MX.MODES, fe
    ok = n_bad == 0
    print(f"[test_metrics] {'PASS' if ok else 'FAIL'}: {n_ep - n_bad}/{n_ep} episodes identical "
          f"({len(files)} result files)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
