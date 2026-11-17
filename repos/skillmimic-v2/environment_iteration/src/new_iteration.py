"""Create the next iteration folder from its parent (LOOP.md step 4). CPU, instant.

    python environment_iteration/src/new_iteration.py fix-2 --summary "one line: what changes" [--from fix-1]

Creates iterations/fix-2/ with:
  iteration.json  copy of the parent's, with name / parent / summary / created set and cfg_env pointing at the copy
                  below ("imported" is dropped: a new iteration is always trained by scripts/train.sh)
  env.yaml        copy of the parent's env config: edit THIS file for the environment changes
  ITERATION.md    template (changes, hypothesis, validation, results, analysis, videos, human notes, decision)
  clips.json      empty clip request (filled after analysis)
Refuses to overwrite an existing iteration. --from defaults to the newest existing iteration.
"""
import argparse
import datetime
import os
import shutil
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import iterlib as L  # noqa: E402

TEMPLATE = """# {name}

Parent: {parent} | Created: {created} | Summary: {summary}
Run: `bash environment_iteration/scripts/train.sh {name}` (human, ~8 h; resumable). Everything below is filled in
by the agent unless marked (human).

## 1. Changes vs {parent}

Each change: what (file / key / class), why (the failure in {parent}'s analysis or video notes it targets),
and how it was checked.

| # | change | where | targets failure | checked by |
|---|---|---|---|---|
| 1 | | | | |

## 2. Hypothesis

Which failure mode / event of {parent} should shrink or move later, and by how much. What result would show the
change did not help.

## 3. Validation before training (agent, each < 30 min)

- [ ] `bash environment_iteration/scripts/check.sh` (unit tests)
- [ ] `bash environment_iteration/scripts/check.sh --gpu {name}` (env builds, 20-epoch train, resume, eval, render)
- [ ] change-specific check (e.g. probe that the new collision/mass/termination behaves as intended)

## 4. Results (from analysis/e{budget}/REPORT.md)

| metric | {parent} | {name} |
|---|---:|---:|
| task success (stoch, 96) | | |
| task success (perturbed) | | |
| paper success | | |
| lift success | | |
| task progress | | |
| fell in clip | | |
| dominant failure (mode @ frame) | | |

## 5. Failure analysis (agent)

## 6. Videos

clips.json -> `video/{name}/...` (see INDEX.md there). Recommended clips and what to look for:

## 7. Video notes (human)

Write what you saw per clip: where the hands are, slipping, one-hand vs two-hand, balance, timing, contact.

## 8. Decision -> next iteration
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    p.add_argument("--from", dest="parent", default="")
    p.add_argument("--summary", required=True)
    a = p.parse_args()
    n = L.iter_number(a.name)
    existing = L.list_iterations()
    parent = a.parent or (existing[-1] if existing else "")
    if not parent:
        sys.exit("no parent iteration exists")
    if a.name in existing or os.path.exists(os.path.join(L.ITER_DIR, a.name)):
        sys.exit(f"{a.name} already exists; never overwrite an iteration (pick fix-{L.iter_number(existing[-1]) + 1})")
    if n <= L.iter_number(parent):
        sys.exit(f"{a.name} must be numbered after its parent {parent}")
    pit = L.load(parent)
    d = os.path.join(L.ITER_DIR, a.name)
    os.makedirs(d)
    src_cfg = os.path.join(L.REPO, pit["cfg_env"])
    dst_cfg = os.path.join(d, "env.yaml")
    shutil.copyfile(src_cfg, dst_cfg)
    it = {k: v for k, v in pit.items() if k != "imported"}
    created = datetime.date.today().isoformat()
    it.update(name=a.name, parent=parent, summary=a.summary, created=created, cfg_env=L.rel(dst_cfg))
    L.write_json(os.path.join(d, "iteration.json"), it)
    with open(os.path.join(d, "ITERATION.md"), "w") as f:
        f.write(TEMPLATE.format(name=a.name, parent=parent, created=created, summary=a.summary,
                                budget=L.protocol()["train"]["epochs"]))
    L.write_json(os.path.join(d, "clips.json"),
                 dict(epoch=L.protocol()["train"]["epochs"],
                      rollouts=[dict(cell="stoch", seed=101, num_envs=8, horizon=180)], clips=[]))
    print(f"[new] {L.rel(d)}/ created from {parent}. Edit env.yaml (and, if needed, src/tasks/ + iteration.json "
          f"'task'), then fill ITERATION.md sections 1-3 and validate (AGENTS.md 5).")


if __name__ == "__main__":
    main()
