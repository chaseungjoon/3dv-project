# v2 변경 및 검증 근거

출발 commit: `d8767fde5a3377001c88b0104b313aa22aa94b01` (이전 실험 도구). 별도 branch `experiments/task-avoidance-v2`. fix task·자산·보상은 고정했다. 본 변경은 실험 실행/계측/검사 도구이며 기존 fix 자체의 과학적 타당성을 인증하지 않는다.

|발견된 문제|변경 위치|확인 방법/남은 한계|
|---|---|---|
|앞 job 성공 후 다음 실행/분석 예외가 COMPLETED로 기록될 수 있음|runner.py::execute_bundle|두 번째 launch 실패·분석 launch 실패·interrupt mock 회귀 검사|
|목표 checkpoint 저장 직후 중단된 run 재개 시 한 epoch 추가 가능|runner.py::branch, branch_entry.py, checkpoints.py|목표 도달 시 audit만 호출하는 CPU 검사, 실제 epoch/frame 검사는 GPU 머신 NOT RUN|
|NaN geometry가 지표/JSON으로 전파|core.py, shared progress runner의 JSON writer|nonfinite geometry/history/action 및 strict JSON 검사|
|lift_attempt가 실패한 접근을 못 잡음|core.py, protocol_v2.json|partial_lift로 재명명, 독립 reach proxy. 정지 손에 움직이는 물체/종료 이후 접근 오판 배제 CPU 검사. 영상 타당화 NOT RUN|
|첫 정책만 반복 검사|runner.py, analyze.py|모든 정책 첫 seed 반복. object뿐 아니라 wrist 차이도 검사. 다른 seed 반복은 미포함|
|없는 primary pair에도 보고서 PASS 가능|analyze.py, runner.py --pair|c0/c1 explicit pair 지원, 누락 pair는 INCONCLUSIVE|
|결과 identity/중복 episode 및 초기 상태 수 검사 부족|analyze.py, core.py|checkpoint hash/seed 확인, duplicate env ID 거부, exact 상태 fingerprint 수|
|남은 수행 능력 검사 없음|runtime.py, evaluate_worker.py, analyze_capability.py, run_capability.sh|실제 donor prefix·self reload·candidate 반복·교체 전 상태 pairing·eligibility. 합성 분석 검사 PASS, GPU NOT RUN|
|실제 training task 경로의 추가 보상/부분 reset 검사 없음|gpu_checks.py, verify_gpu.sh|parent weight0 동등성, buffer/terminal/거리 캐시/history 격리. GPU NOT RUN|

기존 v1 protocol/결과는 보존한다. v2 자동 집계는 회피 가설에 대해 항상 `hypothesis_status: NOT TESTED`를 남긴다. 검사 PASS와 연구 가설을 분리한다.

## 후속 작업

1. 팀원 GPU에서 verify → 두 정책 evaluation → capability 순으로 실행한다.
2. 교체 대조군, 실제 prefix 단계/eligible 수, 반복 오차를 확인한다. 실패 원인을 수정하되 기존 실패 자료를 보존한다.
3. pilot 영상의 수동 판정으로 reach proxy를 검토한다. 자동 영상 녹화와 판정 일치도 집계는 아직 구현하지 않았다.
4. 결과와 비용 검토 후 명시적으로20epoch smoke, 그다음500epoch C0/C1을 실행한다. 여기서는 어느 학습도 실행하지 않았다.
5. 탐색적 결과를 바탕으로 독립 source 학습, held-out 평가, 최소 효과 크기를 정해 확증한다.

기존 progress_reward의 from-scratch launcher는 이번 branch 완료/재개 수정 대상이 아니다. 이번 비교에는 task_avoidance/run_branches.sh를 사용한다. 공유 JSON writer에만 비유한 값 거부를 추가했다.
