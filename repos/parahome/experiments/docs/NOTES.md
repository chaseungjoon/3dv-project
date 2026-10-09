# ParaHome 노트 (2026-10-09)

업스트림: `github.com/snuvclab/ParaHome@535dada` (데이터 + 시각화 코드). 데이터 라이선스 CC BY-NC-SA 4.0.

## 1. 바뀐 것 / 추가한 것

| 항목 | 내용 |
|---|---|
| `pyproject.toml` (uv, Python 3.10) | audit용 CPU 환경: numpy, scipy, trimesh, fast-simplification, matplotlib, gdown. 업스트림 `install.md`의 conda + torch 2.0.1 + open3d는 시각화용이라 설치하지 않았다 |
| `data/` | Google Drive 원본: seq (207개, 6.8 GB), scan (22 물체, 393 MB), smplx_seq (534 MB), metadata, joint_info. 텍스트 annotation 파일 이름은 README의 `text_annotations.json`이 아니라 실제로는 **`text_annotation.json`** |
| `experiments/` | audit, WristMimic 실행/평가 래퍼, 리포트 (업스트림 코드는 그대로) |
| **`../t09-wristmimic/.../sim_object/bookshelf/base_{visual,collision}.obj`** | WristMimic 배포본에 책장 mesh가 없어서 Place Book 장면이 `Failed to resolve collision mesh 'bookshelf/base_collision.obj'` → `KeyError: 'bookshelf_base'`로 죽었다. 다른 물체는 visual = ParaHome 스캔과 **바이트 단위로 같고**, collision = 같은 mesh를 면 14.8%로 줄인 것 (desk 0.1482, book 0.1482, diningtable 0.1492, kettle 0.1416)이라 같은 방법으로 만들었다 (`tools/make_bookshelf_asset.py`, 230,747면, 경계 차이 0.8 mm). T09 저장소의 `.gitignore`에 추가 (115 MB) |

## 2. ParaHome ↔ WristMimic(InterAct) 형식 (확인한 것)

T09 저장소의 `InterAct/Parahome/*.pt`가 원본 ParaHome에서 어떻게 만들어졌는지 주전자 장면으로 역추적했다.
(Place Book 장면이 이미 변환되어 있어서 실제 변환은 하지 않았다. 다른 구간이 필요할 때를 위한 기록.)

| 항목 | 결과 |
|---|---|
| 구간 | `metadata.interval_key` (예: `200 355`). 원본 annotation(`215 360`)과 조금 다르다 |
| 관절 dof (153) | `smplx_seq/<s>/smplx_pose.pkl`의 body_pose(21x3) + hand_pose(30x3) axis-angle을 그대로, 순서만 XML DFS 순으로 바꿈 (최대 오차 0) |
| 루트 | 위치 = `transl` + 피험자별 상수 (SMPL-X 골반 관절, s110: (0.0006, −0.3115, 0.0031)), 회전 = `global_orient` → quaternion (xyzw) |
| 몸 위치/회전 (52) | XML 뼈 offset + axis-angle FK (hinge xyz 순차 FK가 아님). 재현 오차 0.17 mm / 4e-7 |
| 물체 | `object_transformations.pkl`의 `<obj>_base` 그대로 (오차 6e-8) |
| 접촉 라벨 | 몸 52개 x 물체마다 거리 구간 코드 (−1: 멀다, 1/2: 1 cm 이내 ...). SMPL-X mesh 기반으로 보이며 재현하지 않았다 (SMPL-X 모델 라이선스 필요) |

## 3. audit 결과 요약 (`experiments/results/audit/AUDIT.md`)

| 행동 | 구간 수 | 들어올림 | 물체 이동 | 들고 걸은 거리 | 두 손 비율 |
|---|---|---|---|---|---|
| drink_cup | 110 | 0.43 m | 0.06 m | 0.06 m | 0.14 |
| place_book | 89 | 0.20 m | 1.49 m | 0.55 m | 0.85 |
| move_kettle | 57 | 0.31 m | 1.74 m | 0.97 m | 0.16 |

컵 마시기는 "제자리에서 높이 들기", 책/주전자는 "들고 걷기"다. 시뮬레이션 장면 단위로는 컵(s110) 잡기 23 → 들기 47, 책(s11) 16 → 45,
주전자(s110, T09) 26 → 47 frame으로 **잡기에서 들기까지의 시점이 비슷하다** → T09가 막힌 frame 45~60과 같은 구간을 세 장면이 모두 지난다.
