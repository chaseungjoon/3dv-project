#!/usr/bin/env bash
# 체크포인트 하나를 viewer로 재생하고 mp4로 저장한다 (실패 사례 영상용, 화면 필요).
#   사용법: bash experiments/scripts/record_video.sh <checkpoint.pth> [출력이름]
#   결과:   experiments/results/videos/<출력이름>.mp4  (+ 같은 이름의 평가 json/npz)
# 평가 설정과 지표는 결정적 평가와 같다. 카메라는 골반을 따라간다.
# 주의: viewer 창에서 V 키(viewer sync 끄기)를 누르면 시뮬레이션이 멈춘다.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CKPT="${1:?체크포인트 경로가 필요합니다}"
NAME="${2:-$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "${CKPT%.pth}")}"
OUT_DIR="$RESULTS_DIR/videos"
FRAMES="$OUT_DIR/frames_$NAME"
mkdir -p "$OUT_DIR"
rm -rf "$FRAMES"

"$PY" experiments/tools/evaluate.py --checkpoints "$CKPT" --mode det --viewer \
  --image_dir "$FRAMES" --tag "video" --force \
  --motion_file "$SCENE" --robot_type "$ROBOT" || exit 1

ffmpeg -loglevel error -y -framerate 30 -pattern_type glob -i "$FRAMES/rgb_env0_frame*.png" \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "$OUT_DIR/$NAME.mp4" || exit 1
rm -rf "$FRAMES"
t09_log "저장: $OUT_DIR/$NAME.mp4"
