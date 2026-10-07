# T10 CrossFormer baseline: 빠른 시작

과제 **T10 Cross-embodiment robot manipulation**의 baseline인 **CrossFormer** (Doshi et al., CoRL 2024, arXiv 2408.11812)를
공식 130M 체크포인트 그대로(frozen) 두 로봇에서 평가한다. 학습은 하지 않는다 (T10은 inference/test-time만 허용).

- 두 embodiment: **WidowX** (Bridge, 5 Hz)와 **Google Robot** (RT-1/fractal, 3 Hz). 둘 다 학습 mixture에서 가장 큰 단일 팔 데이터 (가중치 각 0.17)이고, SimplerEnv에 closed-loop 시뮬레이터가 있다.
- 오프라인: 학습에 쓰이지 않은 held-out episode에서 action 오차와 실행 제약 위반을 잰다.
- Closed-loop: SimplerEnv(visual matching)에서 task 성공률을 잰다.

자세한 설명은 [README.md](README.md), 규칙과 지표 정의는 [docs/PROTOCOL.md](docs/PROTOCOL.md), 코드 분석과 원본 대비 변경점은 [docs/CODE_NOTES.md](docs/CODE_NOTES.md)에 있다.

> **모든 명령은 저장소 루트 `~/code/3dv-project/repos/t10-crossformer` 에서 실행한다.**
> 긴 단계(Phase 4)는 `tmux` 안에서 돌리는 것을 권장한다. GPU를 쓰는 단계는 **한 번에 하나씩** 돌린다.

```bash
cd ~/code/3dv-project/repos/t10-crossformer
```

---

## 1. 준비 (한 번만, 이 컴퓨터에서는 이미 완료)

```bash
bash experiments/scripts/setup.sh
```

uv 환경(jax 0.6.2 + CUDA 12, TF 2.19), 체크포인트(500 MB), held-out 데이터(약 1.6 GB), SimplerEnv asset(약 400 MB),
문장 인코더 캐시를 받고, 로봇별 action 규약을 측정한다. 다시 실행해도 안전하다 (있는 것은 건너뜀). 이미 다 있으면 약 1분.
**sudo는 필요 없다.**

## 2. 파이프라인 점검 (약 8분, 2026-10-07 통과)

```bash
bash experiments/scripts/pipeline_check.sh
```

모든 단계를 episode 몇 개로 한 번씩 돌린다. 마지막 줄이 `파이프라인 점검 통과` 이면 된다.
결과는 `experiments/results/_check/` 에만 쓰므로 본 결과와 섞이지 않는다 (숫자 자체는 의미 없음).

## 3. 본 실험 (순서대로 복사해서 실행)

| 단계 | 명령 | 시간 (추정) | 무엇을 얻나 |
|---|---|---|---|
| Phase 0 | `bash experiments/scripts/phase0_notebook.sh` | 1분 (완료) | 공식 Colab 예제의 로컬 재현 그림 |
| **Phase 1** | `bash experiments/scripts/phase1_baseline.sh` | 약 10분 | **baseline 표**: 로봇별 action 오차, 제약 위반, 단위 probe |
| Phase 2 | `bash experiments/scripts/phase2_conditioning.sh` | 약 25분 | 언어 vs goal image vs 없음, history 길이 |
| Phase 3 | `bash experiments/scripts/phase3_mismatch.sh` | 약 15분 | 규약 불일치(crop, 좌우 반전, 제어 주기)의 영향 |
| **Phase 4** | `bash experiments/scripts/phase4_closed_loop.sh` | 약 1.5시간 | **closed-loop 성공률** (WidowX 96 + Google Robot 540 episode) |
| (선택) | `SUITES="drawer" bash experiments/scripts/phase4_closed_loop.sh` | 약 1시간 | Google Robot 서랍 열기/닫기 216 episode (ray tracing) |
| (선택) | `VARIANTS="sim_notask" bash experiments/scripts/phase4_closed_loop.sh` | 약 1.5시간 | 언어를 빼면 closed-loop 성공률이 바뀌는가 |

한 번에 Phase 1~4를 이어서 돌리려면:

```bash
bash experiments/scripts/phase1_baseline.sh && \
bash experiments/scripts/phase2_conditioning.sh && \
bash experiments/scripts/phase3_mismatch.sh && \
bash experiments/scripts/phase4_closed_loop.sh
```

시간은 점검에서 잰 속도(오프라인 약 10 ms/window, closed-loop episode당 약 6~10초)로 계산한 추정이다.
각 Phase 스크립트는 끝나면 `report.sh`를 자동으로 부른다.

## 4. 결과 보기

```bash
bash experiments/scripts/report.sh            # 언제든 다시 만들 수 있다 (GPU 안 씀, 1분 이내)
xdg-open experiments/results/report/REPORT.md # 또는 VS Code에서 열기
```

| 파일 | 내용 |
|---|---|
| `experiments/results/report/REPORT.md` | 모든 표 (Phase 1~4, 위반, 단위 probe, paired 차이) + 그림 |
| `experiments/results/report/offline.csv`, `sim.csv` | 발표 표용 원자료 |
| `experiments/results/report/fig_*.png` | 발표용 그림 |
| `experiments/results/conventions/CONVENTIONS.md` | 로봇별 action 규약 (단위, 좌표계 회전, 지연) |
| `experiments/results/notebook/` | Phase 0 (공식 notebook 재현) |
| `experiments/runs/` (git 제외) | 원시 예측(npz), closed-loop 영상(mp4)과 action log |

## 5. 자주 쓰는 옵션

```bash
DATASETS="bridge_dataset fractal20220817_data taco_play" bash experiments/scripts/phase1_baseline.sh  # Franka 추가
SUITES="bridge" bash experiments/scripts/phase4_closed_loop.sh       # WidowX만 (약 15분)
EVAL_BATCH=16 bash experiments/scripts/phase1_baseline.sh            # GPU 메모리가 부족할 때
T10_VERBOSE=1 bash experiments/scripts/phase1_baseline.sh            # 걸러낸 TF/XLA 경고까지 전부 보기
```

## 6. 문제가 생기면

| 증상 | 원인 / 해결 |
|---|---|
| `가상환경 없음`, `체크포인트 없음`, `데이터 없음` | `bash experiments/scripts/setup.sh` |
| `RESOURCE_EXHAUSTED: Out of memory` | 다른 GPU 작업(T09 학습 등)을 끄거나 `EVAL_BATCH=16` |
| `GLFW error: X11: Failed to open display` | 무시해도 된다. SimplerEnv는 화면 없이(headless) 렌더링한다 |
| `TensorFlow was not built with CUDA kernel binaries compatible with compute capability 12.0` | 무시. TF는 CPU에서만 쓴다 (데이터, 문장 인코더) |
| 처음 실행이 1~2분 멈춘 듯 보임 | 모델 로드 + jax 컴파일. 프로세스마다 약 30~50초 |
