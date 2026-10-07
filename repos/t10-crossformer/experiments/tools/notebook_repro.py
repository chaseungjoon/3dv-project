"""Local reproduction of the official inference_pretrained.ipynb (Colab), step 1 and step 2.

    python experiments/tools/notebook_repro.py

Step 1: random images through the bimanual head (shape check, as in the notebook).
Step 2: one held-out Bridge episode through the single_arm head, language-conditioned exactly like the notebook
(window 5, no padding: the first 4 steps are skipped), predicted vs GT plot.
Difference from the notebook: the notebook streams the OXE copy gs://gresearch/robotics/bridge/0.1.0 train[:1];
we use the first episode of our held-out rail bridge_dataset val subset, which is what CrossFormer trained on
(train split) and keeps the evaluation honest.
Writes experiments/results/notebook/{notebook_step2_bridge.png, notebook_repro.json}.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import EXP, RESULTS, hide_tf_gpu, load_episodes  # noqa: E402

WINDOW_SIZE = 5


def main():
    hide_tf_gpu()
    import jax

    from crossformer.model.crossformer_model import CrossFormerModel

    out = RESULTS / "notebook"
    out.mkdir(parents=True, exist_ok=True)
    model = CrossFormerModel.load_pretrained(str(EXP / "checkpoints/crossformer"))

    # step 1 (notebook cell 3)
    img = np.random.randint(0, 255, size=(224, 224, 3))[None, None]
    observation = {"image_high": img, "image_left_wrist": img, "image_right_wrist": img,
                   "timestep_pad_mask": np.array([[True]])}
    task = model.create_tasks(texts=["uncap the pen"])
    action = model.sample_actions(observation, task, head_name="bimanual", rng=jax.random.PRNGKey(0))
    step1 = list(action.shape)
    print("step 1 bimanual action shape", step1)

    # step 2 (notebook cells 7-11) on a held-out episode with an instruction
    ep = next(e for e in load_episodes("bridge_dataset", "1.0.0") if e["language"])
    images, instr = ep["images"], ep["language"]
    task = model.create_tasks(texts=[instr])
    pred, true = [], []
    for step in range(len(images) - (WINDOW_SIZE - 1)):
        obs = {"image_primary": images[step : step + WINDOW_SIZE][None],
               "timestep_pad_mask": np.full((1, WINDOW_SIZE), True, dtype=bool)}
        a = model.sample_actions(obs, task, head_name="single_arm",
                                 unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
                                 rng=jax.random.PRNGKey(0))
        pred.append(np.asarray(a[0]))
        true.append(ep["action"][step + WINDOW_SIZE - 1])
    pred, true = np.array(pred), np.array(true)

    labels = ["x", "y", "z", "yaw", "pitch", "roll", "grasp"]  # the notebook's labels
    strip = np.concatenate(images[::3], axis=1)
    fig, axs = plt.subplot_mosaic([["image"] * 7, labels])
    fig.set_size_inches([30, 7])
    for i, lab in enumerate(labels):
        axs[lab].plot(pred[:, 0, i], label="predicted action")
        axs[lab].plot(true[:, i], label="ground truth")
        axs[lab].set_title(lab)
    axs["image"].imshow(strip)
    axs["image"].set_title(f"held-out bridge_dataset val episode {ep['episode']}: '{instr}'")
    plt.legend()
    fig.savefig(out / "notebook_step2_bridge.png", dpi=80)
    res = {
        "step1_bimanual_action_shape": step1,
        "step2_episode": int(ep["episode"]),
        "step2_instruction": instr,
        "step2_steps": int(len(pred)),
        "step2_mae_per_dim": np.abs(pred[:, 0] - true).mean(0).round(5).tolist(),
        "step2_gripper_acc": float(((pred[:, 0, 6] > 0.5) == (true[:, 6] > 0.5)).mean()),
    }
    (out / "notebook_repro.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    print(f"-> {out}/notebook_step2_bridge.png")


if __name__ == "__main__":
    main()
