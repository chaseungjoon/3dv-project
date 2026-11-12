#!/usr/bin/env bash
# PAPER (ours) vs FIX (fix) on place_book, 3 seeds each, 15,260 epochs (~1e9 samples per run). Plan: PAPER_FIX_FULL_BUDGET.md
#   bash run_paper_fix_full.sh            # all 6 runs (= --plan), ~10.5 days on one RTX 5070. Run inside tmux.
#   bash run_paper_fix_full.sh --s0       # PAPER s0 + FIX s0 only (~3.5 days), then stops by itself
#   bash run_paper_fix_full.sh --s0s1     # + FIX s1 + PAPER s1 (~7 days total), then stops
#   Interrupted? Run the same command again. Same output folders for every scope; a larger scope continues.
# Results: experiments/results_paper_budget/summary.md. Details: experiments/scripts/paper_budget/run_full.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/paper_budget/run_full.sh" "$@"
