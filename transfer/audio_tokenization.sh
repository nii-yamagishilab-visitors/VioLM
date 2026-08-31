#!/usr/bin/env bash

set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR=""
MANIFEST_DIR=""
OUTPUT_DIR=""
CODEC_CHECKPOINT=""
NUM_JOBS=4
BATCH_DURATION=150
DEVICE="auto"
OVERWRITE=()

usage() {
  echo "Usage: bash transfer/audio_tokenization.sh --work-dir DIR --codec-checkpoint FILE [options]"
  echo "Options: --manifest-dir DIR --output-dir DIR --device DEVICE --num-jobs N --batch-duration SECONDS --overwrite"
}

while [[ $# -gt 0 ]]; do
  option="${1//_/-}"
  case "${option}" in
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --codec-checkpoint) CODEC_CHECKPOINT="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --batch-duration) BATCH_DURATION="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --overwrite) OVERWRITE=(--overwrite); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${WORK_DIR}" || -z "${CODEC_CHECKPOINT}" ]]; then
  usage >&2
  exit 2
fi

MANIFEST_DIR="${MANIFEST_DIR:-${WORK_DIR}/manifests}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/audio_tokens}"

cd "${ROOT_DIR}"
python model/tokenizer.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --audio-extractor encodec32-ft \
  --codec-checkpoint "${CODEC_CHECKPOINT}" \
  --dataset-parts train validation test \
  --batch-duration "${BATCH_DURATION}" \
  --device "${DEVICE}" \
  --num-jobs "${NUM_JOBS}" \
  "${OVERWRITE[@]}"

python model/display_manifest_statistics.py --manifest-dir "${OUTPUT_DIR}"
