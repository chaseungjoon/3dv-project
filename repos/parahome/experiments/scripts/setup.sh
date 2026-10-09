#!/usr/bin/env bash
# 한 번만 실행 (다시 실행해도 안전, 있는 것은 건너뜀). sudo 불필요.
#  1) audit용 uv 환경 (CPU)
#  2) ParaHome 데이터 (Google Drive, 압축 약 2.7 GB -> 풀면 약 7.7 GB): seq, scan, smplx_seq, metadata, joint_info
#  3) WristMimic 배포본에 빠진 책장 mesh 생성 + 장면 메타데이터 (experiments/configs/interact_meta.json)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
[[ -x "$PY" ]] || uv sync || exit 1
mkdir -p data
# name  google-drive-id  (README.md의 링크)
while read -r name id; do
  case "$name" in
    seq|scan|smplx_seq) [[ -d "data/$name" ]] && continue
      ph_log "다운로드: $name.zip"; "$PY" -m gdown "$id" -O "data/$name.zip" || exit 1
      ( cd data && unzip -q "$name.zip" && rm "$name.zip" ) || exit 1 ;;
    *) [[ -f "data/$name" ]] && continue
      ph_log "다운로드: $name"; "$PY" -m gdown "$id" -O "data/$name" || exit 1 ;;
  esac
done <<'LIST'
metadata.json 1jPRCsotiep0nElHgyLQNjlkHsWgHbjhi
joint_info.pkl 15fGnZn8o4I2bzQtQF-9MliwxKc2IUdzI
seq 10MYSSM2H7f6g2n9nnXta48qmAhZ7r4yd
scan 1-OuWvVFOFCEhut7J2t1kNbr5jv78QNFP
smplx_seq 1Zzj-umCtpcU4QmI4vSjlPShOSHL5sZMX
LIST
ph_log "데이터: $(ls data/seq | wc -l) sequences, $(ls data/scan | wc -l) objects"
# WristMimic 배포본에 없는 책장 mesh를 ParaHome 스캔에서 만든다 (Place Book 장면에 필요, NOTES.md 1절)
"$PY" experiments/tools/make_bookshelf_asset.py || exit 1
if [[ -x "$T09_PY" && ! -f experiments/configs/interact_meta.json ]]; then
  "$T09_PY" - <<'PYEOF' 2>&1 | grep -v -i warn
import torch, glob, json, os
root = os.path.join(os.environ.get("T09_REPO", "../t09-wristmimic"), "InterAct", "Parahome")
out = {}
for d in sorted(glob.glob(root + "/*")):
    p = glob.glob(d + "/*.pt")[0]; x = torch.load(p, map_location="cpu"); m = x["metadata"]
    out[os.path.basename(d)] = dict(file=os.path.basename(p), scene=m["scene"], interval_key=m["interval_key"],
                                    expression=m["expression"], objects=x["objects"], frames=int(x["data"].shape[0]))
json.dump(out, open("experiments/configs/interact_meta.json", "w"), indent=1)
print(f"[parahome] interact_meta.json: {len(out)} clips")
PYEOF
fi
ph_log "준비 완료"
