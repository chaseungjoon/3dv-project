"""Download + prepare the released HMG benchmark shards for one task (same code path as training).

    python experiments/tools/prepare_data.py --task 02_push_button --data-dir experiments/data/policy_data

Calls scripts/train_policy_example.py:prepare_released_task, so `train_policy_example.py --data-dir <same>`
later finds the projected LeRobot datasets and skips the download.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import train_policy_example as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="02_push_button", choices=T.TASKS if hasattr(T, "TASKS") else None)
ap.add_argument("--data-dir", type=Path, default=Path("experiments/data/policy_data"))
a = ap.parse_args()
t0 = time.time()
root, ids = T.prepare_released_task(a.task, a.data_dir)
print(f"[prep] {a.task}: {len(ids)} shards ready under {root} ({time.time() - t0:.0f}s)")
