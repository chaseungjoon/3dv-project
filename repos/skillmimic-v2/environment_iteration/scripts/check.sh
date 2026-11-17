#!/usr/bin/env bash
# Validation of the loop's code. Agent runs it after every code change (both parts stay well under 30 min).
#
#   bash environment_iteration/scripts/check.sh                 # CPU, ~15 s: syntax, metric regression, protocol, iterations
#   bash environment_iteration/scripts/check.sh --gpu fix-2     # + end-to-end on GPU in a scratch folder, ~5-8 min:
#                                                               #   20-epoch train, Ctrl-C, resume, auto eval/analysis, video
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
FAIL=0
ok()  { printf '[check] PASS  %s\n' "$*"; }
bad() { printf '[check] FAIL  %s\n' "$*"; FAIL=1; }

for f in "$EI"/scripts/*.sh; do bash -n "$f" && continue; bad "bash syntax $f"; done; ok "bash syntax (scripts/*.sh)"
"$PY" -m py_compile "$EI_SRC"/*.py "$EI_SRC"/tasks/*.py "$EI_SRC"/tests/*.py && ok "python syntax (src/)" || bad "python syntax"
"$PY" "$EI_SRC/tests/test_metrics.py" --max 400 | tail -1 | grep -q PASS && ok "metrics == experiments/tools/evaluate.py" \
  || bad "metrics regression (run: $PY $EI_SRC/tests/test_metrics.py)"
"$PY" - <<PYEOF && ok "protocol + iterations load" || bad "protocol / iteration definitions"
import sys; sys.path.insert(0, "$EI_SRC"); import iterlib as L
p = L.protocol(); assert p["train"]["epochs"] in p["eval"]["curve"]["epochs"], "budget epoch must be a curve epoch"
names = L.list_iterations(); assert names, "no iterations"
for n in names:
    it = L.load(n); L.train_args(it)
    if it.get("parent"): assert it["parent"] in names, f"{n}: parent {it['parent']} missing"
    import tasks; assert it["task"] in tasks.REGISTRY or it["task"].startswith("SkillMimic"), f"{n}: task {it['task']}"
print("  iterations:", " ".join(names))
PYEOF
for n in $("$PY" "$EI_SRC/iterlib.py" list | awk '{print $1}'); do
  eval "$("$PY" "$EI_SRC/iterlib.py" paths "$n")"
  [[ -f "$RUN_DIR/fingerprint.json" ]] && { "$PY" "$EI_SRC/fingerprint.py" diff "$n" | sed 's/^/  /'; }
done
"$PY" "$EI_SRC/fingerprint.py" imported fix-1 2>/dev/null | tail -1 | sed 's/^/  /'

if [[ "${1:-}" == --gpu ]]; then
  IT="${2:?usage: check.sh --gpu fix-N}"
  ei_gpu_free_or_exit
  "$PY" "$EI_SRC/tests/pipeline_check.py" "$IT" "${@:3}" || FAIL=1
fi
(( FAIL == 0 )) && echo "[check] ALL PASS" || echo "[check] FAILED"
exit "$FAIL"
