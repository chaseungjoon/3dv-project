# environment_iteration

Fix the SkillMimic-V2 `place_book` environment iteration by iteration until the policy reaches ≥ 70% task success
at a fixed 3,000-epoch budget. **Idea and rules: [LOOP.md](LOOP.md). Where we are: [STATUS.md](STATUS.md).
Rules for LLM agents: [AGENTS.md](AGENTS.md).**

Commands (from `repos/skillmimic-v2/`):

| step | who | command | time |
|---|---|---|---|
| train fix-n (+ auto eval, analysis, status) | human, in tmux | `bash environment_iteration/scripts/train.sh fix-n` | ~8 h |
| ...interrupted? | human | the same command again (continues from the newest checkpoint) | |
| evaluate + analyze again | either | `bash environment_iteration/scripts/eval.sh fix-n` | ~6 min |
| re-analyze from stored traces | either | `bash environment_iteration/scripts/analyze.sh fix-n` | seconds |
| render requested clips | either | `bash environment_iteration/scripts/video.sh fix-n` | ~2 min |
| render an ad hoc window | either | `bash environment_iteration/scripts/video.sh fix-n --cell stoch --seed 101 --window 60:110` | ~1 min |
| start the next iteration | agent | `bash environment_iteration/scripts/new_iteration.sh fix-(n+1) "change"` | instant |
| results table | either | `bash environment_iteration/scripts/status.sh` | instant |
| tests | agent | `bash environment_iteration/scripts/check.sh [--gpu fix-n]` | 15 s / ~6 min |

Where to look after a run: `iterations/fix-n/analysis/e3000/REPORT.md` (numbers, failure modes, learning curve),
`video/fix-n/<rollout>/INDEX.md` (clips and what to look for), `iterations/fix-n/ITERATION.md` (changes and
conclusions). Write your observations from the videos into ITERATION.md §7.

Environment variables (rarely needed): `EI_ALLOW_CHANGED=1` resume although the code changed (train.sh),
`EI_KEEP_ALL=1` keep every checkpoint (~13 GB / iteration), `EI_FORCE_GPU=1` skip the "training is running" guard.
