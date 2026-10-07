"""Held-out episode loader that reproduces CrossFormer's training-time preprocessing.

Per dataset it applies the same standardize_fn (OXE_STANDARDIZATION_TRANSFORMS), takes the same camera as
"image_primary" (OXE_DATASET_CONFIGS), and decodes + resizes with dlimp's Lanczos3 resize to 224x224
(crossformer/data/obs_transforms.py:decode_and_resize). Unlike make_dataset_from_rlds it keeps
the standardized proprio (EEF pose) so execution constraints can be checked, and it does not normalize actions.
"""

import os
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
# pipeline_check.sh points these at runs/_check and results/_check so checks never mix with real results
RUNS = Path(os.environ.get("T10_RUNS", EXP / "runs"))
RESULTS = Path(os.environ.get("T10_RESULTS", EXP / "results"))
RESIZE = (224, 224)


def hide_tf_gpu():
    # TF is only for data/text; it must not take GPU memory from jax
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


def dataset_dir(name: str, version: str) -> Path:
    return DATA / name / version


def load_episodes(
    name: str,
    version: str,
    max_episodes: int = None,
    with_images: bool = True,
    image_mode: str = "squash",
):
    """Yields dicts: episode (int, order in the subset), images (T,224,224,3) uint8, action (T,7) float32
    (standardized, NOT normalized), proprio (T,P) float32, language (str).

    image_mode: "squash" (training preprocessing), "center_crop" (central square, then resize) or
    "hflip" (squash, then mirror)."""
    assert image_mode in ("squash", "center_crop", "hflip"), image_mode
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    from crossformer.data.oxe.oxe_dataset_configs import OXE_DATASET_CONFIGS
    from crossformer.data.oxe.oxe_standardization_transforms import (
        OXE_STANDARDIZATION_TRANSFORMS,
    )

    hide_tf_gpu()
    builder = tfds.builder_from_directory(str(dataset_dir(name, version)))
    # one reader keeps episode order deterministic (= shard order)
    ds = dl.DLataset.from_rlds(builder, split="val", shuffle=False, num_parallel_reads=1)
    standardize = OXE_STANDARDIZATION_TRANSFORMS[name]
    cam = OXE_DATASET_CONFIGS[name]["image_obs_keys"]["primary"]

    def restructure(traj):
        traj = standardize(traj)
        out = {
            "action": tf.cast(traj["action"], tf.float32),
            "proprio": tf.cast(traj["observation"]["proprio"], tf.float32),
            "language": traj["language_instruction"][0]
            if "language_instruction" in traj
            else tf.constant(b""),
        }
        if with_images:
            out["image"] = traj["observation"][cam]
        return out

    def decode(img):
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
        if image_mode == "center_crop":
            h, w = tf.shape(img)[0], tf.shape(img)[1]
            s = tf.minimum(h, w)
            img = tf.image.crop_to_bounding_box(img, (h - s) // 2, (w - s) // 2, s, s)
        img = dl.transforms.resize_image(img, size=RESIZE)
        if image_mode == "hflip":
            img = img[:, ::-1]
        return img

    ds = ds.traj_map(restructure, 1)
    if with_images:
        ds = ds.map(
            lambda t: {**t, "image": tf.map_fn(decode, t["image"], fn_output_signature=tf.uint8)},
            num_parallel_calls=4,
            deterministic=True,
        )
    ds = ds.prefetch(2)
    for i, t in enumerate(ds.as_numpy_iterator()):
        if max_episodes is not None and i >= max_episodes:
            break
        if len(t["action"]) == 0:
            continue
        ep = {
            "episode": i,
            "action": t["action"],
            "proprio": t["proprio"],
            "language": t["language"].decode(),
        }
        if with_images:
            ep["images"] = t["image"]
        yield ep
