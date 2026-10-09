"""Single source of truth for clips and methods (used by train.sh and evaluate.py).

    python experiments/tools/methods.py train_args <clip> <method>   # prints run.py flags for training
    python experiments/tools/methods.py list                         # clips and methods

Methods follow Table 2 of the paper (arXiv 2505.02094). The README has no ParaHome commands, so the
mapping below was read from the code (experiments/docs/CODE_NOTES.md 2).
"""
import os
import sys

# history encoder (experiments/tools/train_hist_encoder.py); SMV2_HIST_CKPT overrides (pipeline check)
HIST_CKPT = os.environ.get("SMV2_HIST_CKPT", "experiments/checkpoints/hist_encoder/parahome_hist60.ckpt")

# clip name = folder under skillmimic/data/motions/ParaHome (the folder name is also the metric key)
CLIPS = {
    "place_book":   dict(cfg="experiments/configs/env/place_book.yaml",   frames=150, obj="book",   static="desk"),
    "drink_cup":    dict(cfg="experiments/configs/env/drink_cup.yaml",    frames=180, obj="cup",    static="diningtable"),
    "place_kettle": dict(cfg="experiments/configs/env/place_kettle.yaml", frames=100, obj="kettle", static="diningtable"),
}

METHODS = {
    # SM + Ours: STF (noisy init + nearest-state resampling), ATS (reweight), HE (history encoder),
    # buffer nodes. No STG on ParaHome (paper 5.2), so no --graph_file / state_switch_prob.
    "ours": dict(
        task="SkillMimicParahomeLocalHistRISBuffernode",
        asset="mjcf/mocap_parahome_boxhand_hist.xml",          # obs +3 (history embedding)
        common=["--hist_length", "60", "--history_embedding_size", "3", "--hist_ckpt", HIST_CKPT],
        train=["--reweight", "--reweight_alpha", "1.0", "--state_init_random_prob", "0.1", "--enable_buffernode"],
    ),
    # SM: SkillMimic (v1) baseline. Random reference state init only.
    "sm": dict(
        task="SkillMimicParahome",
        asset="mjcf/mocap_parahome_boxhand.xml",
        common=[],
        train=[],
    ),
    # SM + T: SM conditioned on the reference phase t/len (repeated 6x in the observation).
    "sm_t": dict(
        task="SkillMimicParahomePhase",
        asset="mjcf/mocap_parahome_boxhand_refobj.xml",        # obs +6 (phase)
        common=[],
        train=[],
    ),
}

TRAIN_CFG = os.environ.get("SMV2_TRAIN_CFG", "experiments/configs/train/parahome.yaml")  # pipeline check: save every 10
EPISODE_LENGTH = 60   # paper Table 8: T = 60


def base_args(clip, method):
    c, m = CLIPS[clip], METHODS[method]
    return ["--task", m["task"], "--cfg_env", c["cfg"], "--cfg_train", TRAIN_CFG,
            "--asset_file_name", m["asset"],
            "--motion_file", f"skillmimic/data/motions/ParaHome/{clip}"] + m["common"]


def train_args(clip, method):
    return base_args(clip, method) + ["--episode_length", str(EPISODE_LENGTH)] + METHODS[method]["train"]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        print("clips:", " ".join(CLIPS)); print("methods:", " ".join(METHODS))
    elif len(sys.argv) == 4 and sys.argv[1] == "train_args":
        if sys.argv[2] not in CLIPS or sys.argv[3] not in METHODS:
            sys.exit(f"unknown clip/method: {sys.argv[2]} {sys.argv[3]}")
        print(" ".join(train_args(sys.argv[2], sys.argv[3])))
    else:
        sys.exit(__doc__)
