# Decisions

- User requested implementation and GitHub upload; local GPU training is excluded.
- Freeze source fix at 71e39eb84149cb1ab8fa191d98cb5d1ea352d2a2. Only new opt-in task registration and method registration modify existing files.
- A/B share instrumentation; A weight0 directly preserves parent reward. Actual same-state equivalence on Isaac Gym is NOT RUN.
- Use fixed final book endpoint first, clip0.05 and pilot weight1. CLI weight overrides JSON's documented default. No cup endpoint reuse.
- No automatic installations, external service runtime dependency, pretrained policy fine-tuning, automatic env-count reduction, or automatic full experiment sweep.
- Resume is explicit full-state continuation from a numbered checkpoint and requires matching source/config/data fingerprint. It is not a claim of bitwise CUDA determinism.
- Main evaluation keeps existing fix protocol, including its initialization noise and differing legacy metric windows. Auxiliary clip metrics are separate.
- GPU elapsed costs are extrapolated from historical runs.csv, not measurements of this package.
- Read original scripts: experiments/tools/methods.py::train_args/base_args; evaluate.py::select_checkpoints/main; skillmimic/env/tasks/skillmimic_parahome_localhist.py::SkillMimicParahomeLocalHistRISBuffernodePhase; learning/skillmimic_agent.py resume/save implementation; experiments/configs/train/parahome.yaml save_frequency=10.
- Original repo push access is READ; deliver through authenticated user's fork, preserve upstream main.
