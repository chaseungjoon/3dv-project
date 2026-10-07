"""Download a held-out subset of an RLDS dataset (a few tfrecord shards) and expose it as split "val".

    python experiments/tools/download_subset.py bridge_dataset fractal20220817_data

Writes experiments/data/<name>/<version>/{dataset_info.json, features.json, <name>-val.tfrecord-*, SUBSET.json}.
Re-running skips files whose size already matches. The held-out rule is in configs/datasets.yaml.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import requests
import yaml

EXP = Path(__file__).resolve().parents[1]


def get(url, **kw):
    r = requests.get(url, timeout=60, **kw)
    r.raise_for_status()
    return r


def remote_size(url):
    r = requests.head(url, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["content-length"])


def fetch(url, dst: Path, size: int):
    if dst.exists() and dst.stat().st_size == size:
        return "skip"
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    if tmp.stat().st_size != size:
        raise IOError(f"size mismatch for {url}: {tmp.stat().st_size} != {size}")
    tmp.rename(dst)
    return "ok"


def pick_shards(spec: str, n_shards: int):
    mode, k = spec.split(":")
    k = int(k)
    if mode == "first":
        return list(range(min(k, n_shards)))
    if mode == "last":
        return list(range(max(0, n_shards - k), n_shards))
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--config", default=str(EXP / "configs/datasets.yaml"))
    ap.add_argument("--out", default=str(EXP / "data"))
    ap.add_argument("--shards", default=None, help="override, e.g. first:1 (pipeline check)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    for name in args.names:
        c = cfg[name]
        base = c["base_url"].rstrip("/")
        info = get(f"{base}/dataset_info.json").json()
        features = get(f"{base}/features.json").content
        split = next(s for s in info["splits"] if s["name"] == c["source_split"])
        lengths = [int(x) for x in split["shardLengths"]]
        n = len(lengths)
        idx = pick_shards(args.shards or c["shards"], n)

        # held-out check: global episode range of the chosen shards within the source split
        starts = [sum(lengths[:i]) for i in range(n)]
        ep_lo, ep_hi = starts[idx[0]], starts[idx[-1]] + lengths[idx[-1]]
        total = sum(lengths)
        if c["heldout"] == "tail95":
            # tfds rounds percent boundaries to the closest example
            boundary = int(round(total * 95 / 100))
            assert ep_lo >= boundary, f"{name}: shards start at episode {ep_lo} < train[95%:] boundary {boundary}"
            note = f"{c['source_split']}[95%:] starts at episode {boundary}; subset = episodes [{ep_lo}, {ep_hi})"
        else:
            assert c["heldout"] == "split"
            note = f"whole '{c['source_split']}' split is unseen in training; subset = episodes [{ep_lo}, {ep_hi})"

        out = Path(args.out) / name / c["version"]
        out.mkdir(parents=True, exist_ok=True)
        k = len(idx)
        jobs = []
        for j, i in enumerate(idx):
            src = f"{base}/{name}-{c['source_split']}.tfrecord-{i:05d}-of-{n:05d}"
            dst = out / f"{name}-val.tfrecord-{j:05d}-of-{k:05d}"
            jobs.append((src, dst))
        with ThreadPoolExecutor(args.workers) as ex:
            sizes = list(ex.map(lambda j: remote_size(j[0]), jobs))
        print(f"[{name}] {k} shards, {sum(sizes) / 1e9:.2f} GB, {sum(lengths[i] for i in idx)} episodes -> {out}")
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda a: fetch(a[0][0], a[0][1], a[1]), zip(jobs, sizes)))
        print(f"[{name}] downloaded {res.count('ok')}, already present {res.count('skip')}")

        # local dataset_info: a single "val" split made of the downloaded shards
        new_split = {
            "name": "val",
            "shardLengths": [str(lengths[i]) for i in idx],
            "numBytes": str(sum(sizes)),
        }
        if "filepathTemplate" in split:
            new_split["filepathTemplate"] = split["filepathTemplate"]
        info["splits"] = [new_split]
        (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
        (out / "features.json").write_bytes(features)
        (out / "SUBSET.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "source": base,
                    "source_split": c["source_split"],
                    "source_shards": idx,
                    "source_num_shards": n,
                    "episodes": sum(lengths[i] for i in idx),
                    "heldout_note": note,
                },
                indent=2,
            )
        )
        print(f"[{name}] {note}")
        # stale shards from a larger earlier download would break the "-of-K" naming
        for f in out.glob(f"{name}-val.tfrecord-*"):
            if not f.name.endswith(f"-of-{k:05d}"):
                f.unlink()


if __name__ == "__main__":
    sys.exit(main())
