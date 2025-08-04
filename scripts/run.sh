#!/usr/bin/env bash
set -euo pipefail

# ---------- Usage help ----------
show_help() {
cat << EOF
Usage: ./run.sh [OPTIONS]

Run multiview reconstruction using StreamVGGT inside a Docker container.

Options:
  --image_dir PATH       Path to image folder (default: ./data/images)
  --checkpoint PATH      Path to checkpoint file (default: ./ckpt/model.pt)
  --output_dir PATH      Path to output folder (default: ./eval_results/mv_recon/custom_dataset)
  -h, --help             Show this help message and exit

Example:
  ./run.sh --image_dir ./data/my_scene --checkpoint ./ckpt/model.pt
EOF
}

# ---------- Default values ----------
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_DIR="$ROOT_DIR/images"
CHECKPOINT="$ROOT_DIR/ckpt/model.pt"
OUTPUT_DIR="$ROOT_DIR/ofolder"

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image_dir)
      IMAGE_DIR="$(realpath "$2")"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$(realpath "$2")"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$(realpath "$2")"
      shift 2
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# ---------- Create output directory ----------
mkdir -p "$OUTPUT_DIR"

# ---------- Run Docker ----------
echo "[INFO] Running StreamVGGT inside Docker..."
echo "       Image dir:    $IMAGE_DIR"
echo "       Checkpoint:   $CHECKPOINT"
echo "       Output dir:   $OUTPUT_DIR"

docker run --rm -it --gpus all \
  -v "$ROOT_DIR":/workspace/StreamVGGT \
  --workdir /workspace/StreamVGGT \
  stream-vggt \
  conda run -n StreamVGGT bash src/eval/mv_recon/run.sh \
    --dataset_dir /workspace/$(realpath --relative-to="$ROOT_DIR" "$IMAGE_DIR") \
    --checkpoint /workspace/$(realpath --relative-to="$ROOT_DIR" "$CHECKPOINT") \
    --output_dir /workspace/$(realpath --relative-to="$ROOT_DIR" "$OUTPUT_DIR")

echo "[INFO] Done. Output written to: $OUTPUT_DIR"
