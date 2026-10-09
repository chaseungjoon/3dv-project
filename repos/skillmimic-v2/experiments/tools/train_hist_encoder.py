"""Train the SkillMimic-V2 history encoder (HE) on ParaHome clips.

The released repo only has a hard-coded script (skillmimic/utils/state_prediction_parahome.py, path
`Sigraph_experiment/app3/parahome_mix`, 3000 epochs, no checkpoint saving). This wraps the same
dataset/model classes with a CLI and writes one Lightning checkpoint that
`skillmimic/utils/history_encoder.py:HistoryEncoder.resume_from_checkpoint` can load (input 394, embedding 3).

    python experiments/tools/train_hist_encoder.py --motion_dir skillmimic/data/motions/ParaHome \
        --out experiments/checkpoints/hist_encoder/parahome_hist60.ckpt
"""
import argparse
import os
import sys
import time

# The upstream module trains at import time (module-level code at the bottom), so import its pieces
# by exec'ing only the definitions.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from isaacgym.torch_utils import *  # noqa: F401,F403  (isaacgym before torch)
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def load_upstream():
    src = open(os.path.join(REPO, "skillmimic/utils/state_prediction_parahome.py")).read()
    src = src.split("# Usage")[0]  # drop the hard-coded training run
    ns = {"__name__": "state_prediction_parahome"}
    exec(compile(src, "state_prediction_parahome.py", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", default="skillmimic/data/motions/ParaHome")
    ap.add_argument("--out", default="experiments/checkpoints/hist_encoder/parahome_hist60.ckpt")
    ap.add_argument("--history_length", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=3000)  # upstream state_prediction_parahome.py
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pl.seed_everything(args.seed)
    ns = load_upstream()
    ds = ns["CustomDataset"](args.motion_dir, args.history_length)
    print(f"[hist] samples={len(ds)} from {len(ds.file_paths)} clips")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = ns["ComprehensiveModel"](args.history_length)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    trainer = pl.Trainer(max_epochs=args.epochs, devices=1, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    t0 = time.time()
    trainer.fit(model, dl)
    trainer.save_checkpoint(args.out)
    loss = trainer.callback_metrics.get("train_loss_epoch")
    print(f"[hist] saved {args.out}  final train_loss={float(loss) if loss is not None else float('nan'):.5f}  "
          f"time={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
