# Result report

Status: PASS / FAIL / NOT RUN / BLOCKED / INCONCLUSIVE (per check; training completion is not research success).

Commit / branch / dirty tree / manifest fingerprint:
GPU / packages / start-end UTC / actual GPU hours / disk:
Experiment ID / weight / train seed / samples / resume checkpoint:
Weight selection protocol and cost (if any):

|Condition/train seed|Final epoch|Evaluation seed / N|Task|Legacy lift|Clip fall|Aux stable lift|Survivors / N|Goal error m (survivors)|
|---|---|---|---|---|---|---|---|---|

Report each independent training seed before mean and dispersion; evaluation episodes are not training replicates. Include denominator, clip/horizon, initialization mode, video IDs and selection rule.

Checks: original fix vs weight0 at identical states; reset/buffer/terminal; finite rewards; scale of base/added; saving; interrupt/resume; evaluation; archive. Mark actual executed checks only.

RQ1 behavioral observations vs Task Avoidance hypothesis:
RQ2 progress effect and stability cost:
RQ3 selected weights and held-out evaluation:
RQ4 independent seed/task repetition:
Unintended throwing/oscillation evidence:
Limitations / missing files / next experiment / allowed conclusion:

Attach experiment.json, run_meta.json, reward_components.jsonl, train.log, gpu.csv, evaluation manifests/logs/JSON/NPZ, CSV/SVG and any videos. Checkpoints optional.
