# 팀원 실행 안내

대상: 기존 RTX 5070 Linux/Isaac Gym 환경. 자동 설치하지 않는다. 작업 기준 커밋은 원 저장소 `71e39eb84149cb1ab8fa191d98cb5d1ea352d2a2`이다. 새 파일이 있는 브랜치를 받아 기존 데이터와 encoder를 연결한다. 기존 작업 트리가 더럽다면 먼저 보존하고 별도 checkout을 사용한다. 기존 runs를 이동하거나 덮어쓰지 않는다.

```bash
cd /path/to/3dv-project/repos/skillmimic-v2
export PY=/absolute/path/to/existing/isaacgym/python
bash experiments/scripts/progress_reward/preflight.sh
bash experiments/scripts/progress_reward/run_compare.sh --stage smoke --epochs 20 --label pilot --dry-run
bash experiments/scripts/progress_reward/run_compare.sh --stage smoke --epochs 20 --label pilot
```

필수 외부 파일: `experiments/checkpoints/hist_encoder/parahome_hist60.ckpt`, `skillmimic/data/motions/ParaHome/s10/object_transformations.pkl`. book motion, URDF, MJCF와 그 mesh 의존성도 필요하다. 전체 저장소와 기존 머신의 자산을 사용한다. Python, CUDA, Isaac Gym을 업그레이드하지 않는다. `PY`는 학습 환경, `ORCHESTRATOR_PY`는 표준 라이브러리 runner용 Python이며 기본 python3이다.

Smoke 두 run에서 NaN, reward_components.jsonl의 base/added 크기, checkpoint 저장, 아래 평가를 확인한다. 별도 smoke label에서 실행 중 Ctrl+C 후 같은 명령에 `--resume`을 붙여 재개도 확인한다. 10 epoch 이전 중단으로 checkpoint가 없으면 새 label로 시작해야 한다. resume은 장애 복구이며 새로운 독립 학습 seed가 아니다.

GPU에서 원 fix와 weight=0의 동일 상태 보상/관측 비교, 부분 reset과 buffer 경계 검증은 아직 NOT RUN이다. 이 확인 없이 본 학습이 검증됐다고 표현하지 않는다.

```bash
# 본 학습: A/B 순차 실행. smoke 성공과 비용 검토 후 팀원이 명시적으로 실행한다.
bash experiments/scripts/progress_reward/run_compare.sh --stage main --epochs 3000 --label main --seeds 0 --weight 1 --execute
# 중단 시 동일 인자에 --resume 추가. 이미 완료된 A는 건너뛰고 B를 재개한다.
bash experiments/scripts/progress_reward/run_compare.sh --stage main --epochs 3000 --label main --seeds 0 --weight 1 --execute --resume
```

가중치 1은 임시 pilot 기본값이다. smoke에서 스케일 확인 후 선택값을 사전에 기록한다. JSON config의 clip이 실제 로드되며 weight는 CLI 값이 우선한다. num-envs를 OOM 대응으로 자동 축소하지 않는다. 변경하려면 별도 label과 동일 A/B 조건을 새로 정한다.

```bash
RUN=experiments/runs/progress_reward/book_main_main_baseline_w0_s0_n2048_e3000_scratch
# B는 book_main_main_progress_w1_s0_n2048_e3000_scratch
bash experiments/scripts/progress_reward/evaluate_experiment.sh --run "$RUN" --label final --seeds 101,102,103 --num-envs 32
bash experiments/scripts/progress_reward/evaluate_experiment.sh --run "$RUN" --label curve --seeds 0 --num-envs 8 --epochs 750,1000,1500,2000,2500,2750,3000
# A/B 각각 평가. 기존 label은 재사용 금지. 평가 실패 후에는 새 label 사용.
python3 experiments/tools/progress_reward/analyze_results.py --input "$RUN/evaluation" --output /tmp/book-baseline-summary
bash experiments/scripts/progress_reward/collect_results.sh --run "$RUN" --output /tmp/book-baseline-results.tar.gz
# 필요할 때만 --checkpoints 추가 (대용량)
```

기존 결과의 CPU 분석:
```bash
python3 experiments/tools/progress_reward/analyze_results.py --input experiments/results/eval --output /tmp/previous-fix-analysis
```
기존 정책 재평가에는 기존 run_dir의 run_meta.json과 nn/*.pth가 필요하다. 위 evaluate wrapper에 기존 run 경로를 지정하면 원래 method=fix를 사용한다. 없는 epoch는 오류로 중단한다.

반환물: A/B 각각 결과 tar.gz, CSV/SVG 분석 디렉터리, RESULT_TEMPLATE 작성본. tar에는 명령/해시/패키지/GPU/시간/종료 코드 manifest, 학습 로그, 보상 성분, GPU 사용량, 평가 JSON/NPZ가 포함된다. checkpoint는 기본 제외한다. 영상 자동 저장은 구현하지 않았다. 영상은 고정 epoch/seed/env ID(0,1,2)를 먼저 선정해 기존 GUI 평가 `--viewer`로 녹화하고 수동 판정표를 함께 전달한다. viewer는 1환경이므로 여러 env ID의 궤적 선택과 동일하다고 간주하지 말고 별도 seed 목록을 사전에 동결한다. 예: `"$PY" experiments/tools/evaluate.py --run_dir "$RUN" --which final --mode det --seed 101 --viewer --out_dir /tmp/book-video-eval`. 좋은 사례로 교체하지 않는다.

학습 COMPLETED와 평가 COMPLETED는 분리된다. lock은 이 checkout 내부 동시 실행만 방지하므로 다른 checkout의 GPU 작업과 일정 조율이 필요하다. 비용 추정은 본 실험 A/B 약 16.2 GPU시간 + 시작/평가 시간, 최소 40 GiB 여유 디스크이다.
