#!/usr/bin/env bash
# T10 준비를 한 번에: uv 환경, CrossFormer 체크포인트(500MB), held-out 데이터 subset(약 1.6GB),
# SimplerEnv asset(약 400MB), 문장 인코더 캐시, embodiment별 action 규약 측정.
# 여러 번 실행해도 안전하다 (이미 있는 것은 건너뜀). 약 10~15분 (네트워크 속도에 따라).
#   사용법: bash experiments/scripts/setup.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
set -e

MS2_COMMIT=ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682   # pyproject.toml의 mani-skill2-real2sim과 같은 commit
OCTO_COMMIT=653c54acde686fde619855f2eac0dd6edad7116b  # SimplerEnv가 논문 값에 쓴 Octo 1.0

t10_log "1/6 uv sync"
uv sync

t10_log "2/6 CrossFormer 체크포인트 (hf://rail-berkeley/crossformer -> experiments/checkpoints/crossformer)"
"$PY" -c "import huggingface_hub as h; h.snapshot_download('rail-berkeley/crossformer', local_dir='experiments/checkpoints/crossformer')"

t10_log "3/6 held-out 데이터 subset (configs/datasets.yaml)"
"$PY" experiments/tools/download_subset.py bridge_dataset fractal20220817_data taco_play

t10_log "4/6 SimplerEnv asset (ManiSkill2_real2sim@${MS2_COMMIT:0:7})"
TP=experiments/third_party/ManiSkill2_real2sim
if [[ ! -d "$TP/data/real_inpainting" ]]; then
  rm -rf "$TP"; mkdir -p "$TP"
  git -C "$TP" init -q
  git -C "$TP" remote add origin https://github.com/simpler-env/ManiSkill2_real2sim
  git -C "$TP" fetch -q --depth 1 origin "$MS2_COMMIT"
  git -C "$TP" checkout -q FETCH_HEAD
fi

t10_log "4b/6 Octo 1.0 (시뮬레이터 검증용 대조 정책, Octo@${OCTO_COMMIT:0:7} + jax 0.6 API 이름 변경)"
OC=experiments/third_party/octo
if [[ ! -f "$OC/octo/model/octo_model.py" ]]; then
  rm -rf "$OC"; mkdir -p "$OC"
  git -C "$OC" init -q
  git -C "$OC" remote add origin https://github.com/octo-models/octo
  git -C "$OC" fetch -q --depth 1 origin "$OCTO_COMMIT"
  git -C "$OC" checkout -q FETCH_HEAD
  grep -rlZ "jax.random.KeyArray\|jax.tree_map\|jax.tree_leaves" "$OC/octo" | xargs -0 sed -i \
    -e 's/jax\.random\.KeyArray/jax.Array/g' -e 's/jax\.tree_map(/jax.tree.map(/g' -e 's/jax\.tree_leaves(/jax.tree.leaves(/g'
fi

t10_log "5/6 모델 로드 + 문장 인코더(Universal Sentence Encoder) 캐시"
"$PY" - <<'PY' 2>&1 | t10_filter
import sys; sys.path.insert(0, "experiments/tools")
from data import hide_tf_gpu; hide_tf_gpu()
from crossformer.model.crossformer_model import CrossFormerModel
m = CrossFormerModel.load_pretrained("experiments/checkpoints/crossformer")
print("model ok:", sum(x.size for x in __import__("jax").tree.leaves(m.params)) / 1e6, "M params")
PY

t10_log "6/6 embodiment별 action 규약 측정 -> experiments/results/conventions/"
"$PY" experiments/tools/conventions.py bridge_dataset fractal20220817_data taco_play 2>&1 | t10_filter

t10_log "준비 완료. 다음: bash experiments/scripts/pipeline_check.sh"
