"""Add per-episode stage stats (grasped, moved correct object, ...) to closed-loop results that were produced
before sim_eval.py recorded them, by parsing the video names in runs/sim/<variant>/<suite>/videos.

    python experiments/tools/sim_backfill_stages.py sim_baseline
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import RESULTS, RUNS  # noqa: E402
from sim_eval import episode_stats, stage_rates  # noqa: E402


def task_of(video, job):
    """Match a video to its job via env name, build kwargs and robot init (all are part of the video path)."""
    p = str(video)
    if f"/{job['env_name']}" not in p:
        return False
    if any(f"{k}_{v}" not in p for k, v in job["build_kwargs"].items()):
        return False
    return f"rob_{job['robot_init'][0]}_{job['robot_init'][1]}_" in p


def main():
    for variant in sys.argv[1:]:
        for res_path in sorted((RESULTS / "sim" / variant).glob("*.json")):
            r = json.loads(res_path.read_text())
            videos = sorted((RUNS / "sim" / variant / r["suite"] / "videos").rglob("*.mp4"))
            per_task = {}
            for job in r["jobs"]:
                vs = [v for v in videos if task_of(v, job)]
                assert len(vs) == len(job["success"]), (res_path, job["env_name"], len(vs), len(job["success"]))
                job["episode_stats"] = [episode_stats(v) for v in vs]
                assert sum(s["success"] for s in job["episode_stats"]) == sum(job["success"])
                per_task.setdefault(job["task"], []).extend(job["episode_stats"])
            for task, st in per_task.items():
                r["per_task"][task]["stages"] = stage_rates(st)
            res_path.write_text(json.dumps(r, indent=2))
            print(res_path, {t: {k: round(v, 3) for k, v in r["per_task"][t]["stages"].items()} for t in r["per_task"]})


if __name__ == "__main__":
    main()
