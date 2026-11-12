#!/usr/bin/env bash
# PAPER (ours) vs FIX (fix) on place_book, 3 seeds each, 15,260 epochs (~1e9 samples per run). Plan: PAPER_FIX_FULL_BUDGET.md
#   bash run_paper_fix_full.sh     # inside tmux. ~10.5 days on one RTX 5070. Interrupted? Run the same command again.
# Results: experiments/results_paper_budget/summary.md. Details: experiments/scripts/paper_budget/run_full.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/experiments/scripts/paper_budget/run_full.sh" "$@"
