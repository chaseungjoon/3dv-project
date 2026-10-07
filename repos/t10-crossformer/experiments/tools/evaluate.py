"""Open-loop inference of the frozen CrossFormer policy on held-out episodes.

    python experiments/tools/evaluate.py --variant baseline_goal --datasets bridge_dataset fractal20220817_data

For every timestep t of every episode it builds the training-format input (history window ending at t, first
frame repeated with timestep_pad_mask=False before the episode start), runs the single_arm head once, and stores
the NORMALIZED 4-step action chunk plus the ground-truth chunk (raw, standardized units) and proprio.
Writes experiments/runs/<variant>/<dataset>/{pred.npz, meta.json}.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXP, RUNS, hide_tf_gpu, load_episodes  # noqa: E402

HEAD = "single_arm"
HORIZON = 4  # single_arm head action_horizon


class GpuMonitor(threading.Thread):
    """Peak GPU memory via nvidia-smi (whole device, includes the desktop)."""

    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop = every, 0, False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
                )
                self.peak = max(self.peak, int(out.split()[0]))
            except Exception:
                pass
            time.sleep(self.every)


def subsample(ep, k):
    """Run the episode at 1/k of the control rate: keep every k-th frame; the GT action for a kept frame is the
    sum of the k original movement actions (approximate for rotations) and the last gripper command."""
    if k == 1:
        return ep
    T = len(ep["action"])
    idx = np.arange(0, T, k)
    act = []
    for i in idx:
        a = ep["action"][i : i + k]
        act.append(np.concatenate([a[:, :6].sum(0), a[-1:, 6]]))
    out = dict(ep)
    out["action"] = np.asarray(act, np.float32)
    out["proprio"] = ep["proprio"][idx]
    out["images"] = ep["images"][idx]
    return out


class TaskBuilder:
    """Builds task dicts that match either training (zero language when a goal image is used) or the notebook."""

    def __init__(self, model, mode, goal_offset=8):
        self.model, self.mode, self.goal_offset = model, mode, goal_offset
        base = model.create_tasks(goals={"image_primary": np.zeros((1, 224, 224, 3), np.uint8)})
        self.base = {k: v for k, v in base.items() if k != "pad_mask_dict"}
        self.base_pad = dict(base["pad_mask_dict"])
        self.lang_cache = {}
        self.empty_lang = np.asarray(base["language_instruction"][0])  # = USE("")

    def lang_embedding(self, text):
        if text not in self.lang_cache:
            self.lang_cache[text] = np.asarray(self.model.create_tasks(texts=[text])["language_instruction"][0])
        return self.lang_cache[text]

    def for_window(self, ep, t):
        """Returns (goal_image or None, language embedding, goal_pad, lang_pad) for window ending at t."""
        T = len(ep["action"])
        zeros_lang = np.zeros_like(self.empty_lang)
        if self.mode == "lang":
            return None, self.lang_embedding(ep["language"]), False, True
        if self.mode == "goal_final":
            return ep["images"][T - 1], zeros_lang, True, False
        if self.mode == "goal_final_nb":
            return ep["images"][T - 1], self.empty_lang, True, False
        if self.mode == "goal_sub":
            return ep["images"][min(t + self.goal_offset, T - 1)], zeros_lang, True, False
        if self.mode == "none":
            return None, zeros_lang, False, False
        raise ValueError(self.mode)

    def batch(self, items):
        B = len(items)
        tasks = {k: np.repeat(np.asarray(v), B, axis=0) for k, v in self.base.items()}
        pad = {k: np.repeat(np.asarray(v), B, axis=0) for k, v in self.base_pad.items()}
        goal = np.zeros((B, 224, 224, 3), np.uint8)
        for i, (g, lang, gpad, lpad) in enumerate(items):
            if g is not None:
                goal[i] = g
            tasks["language_instruction"][i] = lang
            pad["image_primary"][i] = gpad
            pad["language_instruction"][i] = lpad
        tasks["image_primary"] = goal
        tasks["pad_mask_dict"] = pad
        return tasks


def windows(ep, window):
    """Yields (t, frames[window], pad_mask[window]) with the first frame repeated before the episode start."""
    T = len(ep["action"])
    for t in range(T):
        idx = np.arange(t - window + 1, t + 1)
        mask = idx >= 0
        yield t, ep["images"][np.maximum(idx, 0)], mask


def gt_chunk(ep, t):
    T = len(ep["action"])
    idx = np.arange(t, t + HORIZON)
    valid = idx < T
    return ep["action"][np.minimum(idx, T - 1)], valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--datasets", nargs="+", default=["bridge_dataset", "fractal20220817_data"])
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=int(os.environ.get("EVAL_BATCH", 32)))
    ap.add_argument("--checkpoint", default=str(EXP / "checkpoints/crossformer"))
    ap.add_argument("--out", default=str(RUNS))
    ap.add_argument("--tag", default="", help="suffix for the run dir (e.g. _check)")
    args = ap.parse_args()

    variants = yaml.safe_load(open(EXP / "configs/variants.yaml"))
    dsets = yaml.safe_load(open(EXP / "configs/datasets.yaml"))
    v = variants[args.variant]

    hide_tf_gpu()
    import jax

    from crossformer.model.crossformer_model import CrossFormerModel

    mon = GpuMonitor()
    mon.start()
    t0 = time.time()
    model = CrossFormerModel.load_pretrained(args.checkpoint)
    load_s = time.time() - t0
    tb = TaskBuilder(model, v["task"], v.get("goal_offset", 8))
    rng = jax.random.PRNGKey(0)  # the L1 head is deterministic; rng is unused but required

    for name in args.datasets:
        dc = dsets[name]
        out = Path(args.out) / (args.variant + args.tag) / name
        out.mkdir(parents=True, exist_ok=True)
        B, W = args.batch, v["window"]
        buf_obs, buf_task, buf_key = [], [], []
        rec = {k: [] for k in ("episode", "t", "pred_norm", "gt", "gt_valid", "proprio", "proprio_next", "T")}
        languages = {}
        n_windows, t_infer, compiled = 0, 0.0, False

        def flush():
            nonlocal n_windows, t_infer, compiled
            n = len(buf_obs)
            if n == 0:
                return
            pad_n = B - n  # fixed batch shape -> one compilation
            frames = np.stack([o[0] for o in buf_obs] + [buf_obs[-1][0]] * pad_n)
            masks = np.stack([o[1] for o in buf_obs] + [buf_obs[-1][1]] * pad_n)
            tasks = tb.batch(buf_task + [buf_task[-1]] * pad_n)
            obs = {"image_primary": frames, "timestep_pad_mask": masks}
            ts = time.time()
            a = model.sample_actions(obs, tasks, head_name=HEAD, rng=rng)
            a = np.asarray(a)[:n, :, :7]
            if compiled:
                t_infer += time.time() - ts
            compiled = True
            rec["pred_norm"].append(a)
            n_windows += n
            buf_obs.clear(), buf_task.clear(), buf_key.clear()

        t_start = time.time()
        n_eps, skipped_lang = 0, 0
        for ep in load_episodes(name, dc["version"], args.max_episodes, image_mode=v["image"]):
            # one fixed episode set for all variants: episodes without an instruction are always skipped
            if not ep["language"]:
                skipped_lang += 1
                continue
            ep = subsample(ep, v["frame_skip"])
            T = len(ep["action"])
            languages[ep["episode"]] = ep["language"]
            for t, frames, mask in windows(ep, W):
                gt, valid = gt_chunk(ep, t)
                rec["episode"].append(ep["episode"])
                rec["t"].append(t)
                rec["T"].append(T)
                rec["gt"].append(gt)
                rec["gt_valid"].append(valid)
                rec["proprio"].append(ep["proprio"][t])
                rec["proprio_next"].append(ep["proprio"][min(t + 1, T - 1)])
                buf_obs.append((frames, mask))
                buf_task.append(tb.for_window(ep, t))
                if len(buf_obs) == B:
                    flush()
            n_eps += 1
        flush()
        wall = time.time() - t_start

        np.savez_compressed(
            out / "pred.npz",
            episode=np.asarray(rec["episode"], np.int32),
            t=np.asarray(rec["t"], np.int32),
            T=np.asarray(rec["T"], np.int32),
            pred_norm=np.concatenate(rec["pred_norm"]).astype(np.float32),
            gt=np.asarray(rec["gt"], np.float32),
            gt_valid=np.asarray(rec["gt_valid"], bool),
            proprio=np.asarray(rec["proprio"], np.float32),
            proprio_next=np.asarray(rec["proprio_next"], np.float32),
        )
        meta = {
            "variant": args.variant + args.tag,
            "config": v,
            "dataset": name,
            "embodiment": dc["embodiment"],
            "control_hz": dc["control_hz"] / v["frame_skip"],
            "episodes": n_eps,
            "episodes_skipped_no_language": skipped_lang,
            "windows": n_windows,
            "batch": B,
            "model_load_s": round(load_s, 1),
            "wall_s": round(wall, 1),
            "infer_ms_per_window": round(1000 * t_infer / max(1, n_windows - B), 2),
            "peak_gpu_mem_mib": mon.peak,
            "languages": {str(k): s for k, s in languages.items()},
            "jax": jax.__version__,
            "device": str(jax.devices()[0]),
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"[{args.variant}{args.tag}] {name}: {n_eps} episodes, {n_windows} windows, "
            f"{wall:.0f}s, {meta['infer_ms_per_window']} ms/window, peak GPU {mon.peak} MiB -> {out}"
        )
    mon.stop = True


if __name__ == "__main__":
    main()
