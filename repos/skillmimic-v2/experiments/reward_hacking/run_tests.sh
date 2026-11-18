#!/usr/bin/env bash
# Run several reward-hacking tests in a row (README.md in this folder). Stops at the first test that fails.
#   bash experiments/reward_hacking/run_tests.sh            # 0 1 3 4 5: everything that needs no screen, about 1.5-2 h
#   bash experiments/reward_hacking/run_tests.sh 1 5        # chosen tests
# Test 6 needs you at the screen (viewer) and test 7 takes hours per run, so they are not in the default list.
# Interrupted? Run the same command again: finished evaluations are skipped.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
TESTS=("$@"); (( ${#TESTS[@]} )) || TESTS=(0 1 3 4 5)
for t in "${TESTS[@]}"; do
  [[ -f "test_$t.sh" ]] || { echo "no test_$t.sh (test 2 is part of test 1)"; exit 2; }
  [[ "$t" == 7 ]] && { echo "test 7 needs a variant: bash experiments/reward_hacking/test_7.sh seed1|objw|all"; exit 2; }
  bash "test_$t.sh" || { echo "test_$t.sh failed (exit $?). Fix and rerun: bash experiments/reward_hacking/run_tests.sh ${TESTS[*]}"; exit 1; }
done
echo "all done: ${TESTS[*]}. Results in experiments/results/reward_hacking/"
