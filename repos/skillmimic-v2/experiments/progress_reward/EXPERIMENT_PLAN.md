# 동일 fix에서 기존 보상과 Object Progress 추가 보상 비교

원 fix는 논문 재현 검증 완료 환경이 아니라 고정된 자체 기준이다. 환경 V&V와 기존 수정별 기여 분석은 이번 범위 밖이다. ParaHome에서는 STG 사용을 주장하지 않는다. Task Avoidance는 가설이며 성공률/학습 보상 두 곡선만으로 원인을 확정하지 않는다.

|실험|질문·비교|학습량 / run|평가|비용·다음 단계|
|---|---|---|---|---|
|EXP-01|RQ1 기존 전 체크포인트 행동 변화|새 학습 없음|750/1000/1500/2000/2500/2750/3000 후보 JSON/NPZ; 정책 존재 시 재평가|CPU 분석, 재평가 시간은 실측 필요. 보상과 조작의 관계를 기술|
|EXP-02|추가 코드·저장·재개·평가 연결|A/B 각 20 epoch, 1,310,720 samples|최종, seed101/102/103 ×32|학습 비례추정 합계 약6.5분+시작/평가. NaN 없음·weight0 동등성·buffer/reset 검사 후 진행|
|EXP-03|RQ2 progress가 조작/안정성에 미치는 효과|A/B seed0 각3000, 196,608,000 samples|최종 seed101/102/103 ×32, 곡선 seed0 ×8|합계16.2h+평가, 약27.6GiB checkpoint, 최소40GiB 확보|
|EXP-04|RQ3 가중치 효과|낮음/중간/높음 후보를 smoke 성분 확인 후 동결|선택 seed11/12/13; 최종101/102/103 별도|추가 가중치마다8.1h/13.8GiB 추정. 선택비용 공개|
|EXP-05|RQ4 독립 반복|고정 A/B seeds0,1,2 총6run|조건별 동일 최종 평가|총48.6h/82.8GiB, EXP03 포함. seed0 선택 사용 시 별도 표시|
|EXP-06|견고성·컵 재검증|책은 고정 정책 평가; 컵은 새 목표 설계 후 별도 A/B 학습|공통 교란 seed201/202/203 ×32|평가 실측 후 예산 산정; 컵 비용 미정|

시간 근거: `experiments/results/tables/runs.csv` book 3000epoch 8.10h, epoch9.70초, checkpoint13.81GiB. 추정이며 이번 실행 실측이 아니다. 평가 파일은 수십 MB 수준부터 늘어나므로 실제 생성량 확인. 최소 EXP01→02→03, 권장 가중치 선택 및 독립 반복, 확장 단순 위치보상 강화 대조군/교란/컵. 확장 보상과 컵 학습은 구현하지 않았으며 자동 실행하지 않는다.

## 보상 정의와 대안

A(구현): 단일 book 시연의 마지막 처리된 object position g를 고정 목표로 한다. B(후속): 참조 경로의 앞선 지점을 목표로 하되 한 transition 내 동일 목표를 써야 한다. B는 lookahead와 위상 의존 변수를 추가하므로 첫 비교에서는 A를 사용한다.

월드 좌표 meter, L=||ref[-1]-ref[0]||. L<0.05m 또는 여러 clip이면 거부한다. 추가 보상은 `w * clip((||x_before-g|| - ||x_after-g||)/L, -0.05, 0.05)`이다. before는 pre_physics_step의 target root, after는 부모 reward 계산 직후의 갱신된 target root. buffer 활성 transition은 0, terminal transition은 reset 전 포함. 매 transition 새 before를 기록하여 reset 점프를 보상하지 않는다. 부모 reward 계산 후 합하며 w=0은 아예 덧셈하지 않는다. 제어32step마다 base/raw/absolute raw/added/total 평균을 기록한다.

목표는 책의 최종 시연 위치이며, '내려놓기 완료'라는 의미를 부여하지 않는다. 실제 처리된 endpoint는 GPU goal 로그와 시연 영상으로 확인해야 한다. 원 motion 파일을 로컬 torch로 로드하지 못했으므로 끝 동작의 의미는 INCONCLUSIVE이다. 왕복 signed reward는 이상적 비할인·비clip 조건에서 상쇄되지만 clipping/discount/종료로 exploitable할 수 있다. 던지기·왕복을 영상과 궤적으로 확인하며 정책 불변성을 주장하지 않는다.

물리/형상/물성/history/phase/STF/ATS/buffer/종료/PPO는 원 fix 경로 그대로다. ATS 알고리즘은 같아도 변경 reward가 sampling 통계에 영향을 줄 수 있으며 동일한 sampling 순서를 보장하지 않는다. 최종 적용 CLI 및 모든 추적 파일/자산/data/encoder SHA256을 manifest에 기록한다. runtime 물성 검증을 수행했다는 의미는 아니다.

## 평가 규칙 (결과 전 고정)

주 결과: 3000 final checkpoint. A/B 같은 학습 samples·seed·num-envs·평가 모드/seed 사용. 최종 평가 seed101/102/103, 각각32환경. 독립 학습 seed와 평가 seed를 구별하며 반복 환경들을 독립 학습 반복으로 세지 않는다. 보상함수가 다른 A/B의 total reward 크기를 직접 비교하지 않는다. 보조 best checkpoint는 선택 seed에서 task_success 최대→동률 stable_lift_clip 최대→동률 더 이른 epoch 규칙; 최종 seed로 고르지 않는다.

기존 evaluate.py 보존: lift/held는 기본300 control steps, task/fall_in_clip은 참조 clip 구간. task는 clip 동안 object error<0.1m와 no fall 조건. det는 action 결정성이고 내부 초기화 noise는 YAML 그대로다. 보조 stable_lift_clip은 같은 episode/clip 구간의 lift와 no-fall을 동시에 요구하며 기존 전체 task 성공과 다르다. 목표 오차는 생존 episode 조건부이므로 반드시 생존 수 병기. 손목 거리는 접촉이 아니다. 실패 frame은 10cm tracking/alive threshold 진단이며 실패 원인 추론은 아니다.

교란은 기존 평가기의 xy radius uniform[0,0.10]m, angle uniform[0,2π], yaw uniform[-π/4,π/4]를 A/B 동일 seed로 적용한다 (`--perturb`). 영역 면적에 균일한 분포는 아니다. no perturb와 별도 label로 기록한다.

EXP01은 저장 JSON/NPZ 우선; 현재 분석기는 원 reward 학습 curve를 자동 복원하지 않는다. 기존 표/원 train log를 epoch 기준으로 대조하고 로그가 없으면 보상-성공 상관은 NOT RUN으로 남긴다. 1000/3000만 선택하지 않는다.

## 결과에 따른 다음 연구

- 조작 개선·안정성 유지: 독립 seed→단순 기존 위치보상 강화 대조군→교란→컵.
- 조작 개선·fall 증가: 가중치와 실패 시점 분석; 새 안정성 보상 즉시 추가하지 않음.
- 차이 불명확: 보상 크기/활성 buffer 구간/목표/학습량/seed 변동 확인.
- 특정 seed만 개선: 추가 독립 학습, learning curve 분석.
- 던지기·왕복: 실패 기록 보존, 새 reward 버전으로 분리.
- 단순 대조군과 차이 없음: progress 설계 고유 효과 미입증.
- 책에서만 개선: 과제 의존성 및 컵 목표 구조 분석. 컵 새 학습은 zero-shot이 아니다.
