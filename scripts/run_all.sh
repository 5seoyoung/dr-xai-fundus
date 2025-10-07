#!/usr/bin/env bash
set -e

CFG=configs/aptos.yaml

# 1) 학습 (간단 예시) — 캐글/로컬 모두 동일 CLI
python -m src.train --config $CFG

# 2) 평가 & 커브/메트릭 저장
python -m src.evaluate --config $CFG --ckpt runs/effb0_best.pt --out-dir artifacts/figs

# 3) CAM 보드 (df_val 없어도 혼합 12 생성됨)
python scripts/make_cam_board.py \
  --cams_dir artifacts/cams_resume \
  --aptos_images /data/aptos2019/train_images \
  --out_jpg artifacts/figs/board_mixed.jpg
