#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_BIN="${MAIN_BIN:-$ROOT_DIR/rgbd_tum}"
VOCAB_FILE="${VOCAB_FILE:-$ROOT_DIR/Vocabulary/ORBvoc.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output_data/tum}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-5}"
RUN_HEARTBEAT_SEC="${RUN_HEARTBEAT_SEC:-30}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
ALLOW_EXIT139_WITH_ARTIFACTS="${ALLOW_EXIT139_WITH_ARTIFACTS:-1}"

# Dataset setup.
# Override TUM_PACKAGE_DIR / SETTINGS_FILE / SEQUENCE_BASE_DIR when needed.
TUM_PACKAGE_DIR="${TUM_PACKAGE_DIR:-$ROOT_DIR/input_data}"
SEQUENCE_BASE_DIR="${SEQUENCE_BASE_DIR:-$TUM_PACKAGE_DIR}"
SETTINGS_FILE="${SETTINGS_FILE:-$TUM_PACKAGE_DIR/TUM1.yaml}"

SEQUENCES=(
  "rgbd_dataset_freiburg1_desk"
  "rgbd_dataset_freiburg1_desk2"
  "rgbd_dataset_freiburg1_room"
  "rgbd_dataset_freiburg1_xyz"
  "rgbd_dataset_freiburg2_desk"
  "rgbd_dataset_freiburg2_xyz"
  "rgbd_dataset_freiburg3_long_office_household"
)

if [[ ! -x "$MAIN_BIN" ]]; then
  echo "Error: executable not found or not executable: $MAIN_BIN"
  echo "Build first (e.g. ./build.sh)."
  exit 1
fi

if [[ ! -f "$VOCAB_FILE" ]]; then
  echo "Error: vocabulary file not found: $VOCAB_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "Warning: SETTINGS_FILE not found at $SETTINGS_FILE (continuing; rgbd_tum may fail if this is required)."
fi

TOTAL_RUNS=$(( ${#SEQUENCES[@]} * RUNS_PER_CONFIG ))
CURRENT_RUN=0

echo "Starting TUM RGB-D evaluation"
echo "Root:        $ROOT_DIR"
echo "Binary:      $MAIN_BIN"
echo "Vocabulary:  $VOCAB_FILE"
echo "Settings:    $SETTINGS_FILE"
echo "Output:      $OUTPUT_ROOT"
echo "Total runs:  $TOTAL_RUNS"
echo

for sequence in "${SEQUENCES[@]}"; do
  seq_path="${SEQUENCE_BASE_DIR}/${sequence}"
  assoc_path="${seq_path}/association.txt"

  if [[ ! -d "$seq_path" ]]; then
    echo "Warning: sequence directory not found, skipping: $seq_path"
    continue
  fi
  if [[ ! -f "$assoc_path" ]]; then
    echo "Warning: association file not found, skipping: $assoc_path"
    continue
  fi

  for ((run_idx=1; run_idx<=RUNS_PER_CONFIG; run_idx++)); do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    run_dir="${OUTPUT_ROOT}/${sequence}/run_${run_idx}"
    mkdir -p "$run_dir"

    rm -f "$ROOT_DIR/CameraTrajectory.txt" "$ROOT_DIR/KeyFrameTrajectory.txt"

    echo "[$CURRENT_RUN/$TOTAL_RUNS] sequence=${sequence}, run=${run_idx}"
    {
      echo "  settings_file=$SETTINGS_FILE"
      echo "  vocab_file=$VOCAB_FILE"
      echo "  heartbeat_sec=$RUN_HEARTBEAT_SEC, timeout_sec=$RUN_TIMEOUT_SEC"
      echo "  launching rgbd_tum..."
    } | tee "${run_dir}/run_meta.txt"

    run_start_epoch="$(date +%s)"
    "$MAIN_BIN" \
      "$VOCAB_FILE" \
      "$SETTINGS_FILE" \
      "$seq_path" \
      "$assoc_path" \
      > "${run_dir}/stdout.log" \
      2> "${run_dir}/stderr.log" &
    run_pid=$!
    timed_out=0
    while kill -0 "$run_pid" 2>/dev/null; do
      sleep "$RUN_HEARTBEAT_SEC"
      now_epoch="$(date +%s)"
      elapsed_sec=$(( now_epoch - run_start_epoch ))
      echo "  still running... elapsed=${elapsed_sec}s pid=${run_pid}"
      if [[ "$RUN_TIMEOUT_SEC" -gt 0 && "$elapsed_sec" -ge "$RUN_TIMEOUT_SEC" ]]; then
        echo "  timeout reached (${RUN_TIMEOUT_SEC}s), stopping pid=${run_pid}"
        kill -TERM "$run_pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$run_pid" 2>/dev/null || true
        timed_out=1
        break
      fi
    done
    if wait "$run_pid"; then
      exit_code=0
    else
      exit_code=$?
    fi
    total_elapsed_sec=$(( $(date +%s) - run_start_epoch ))
    echo "  finished run in ${total_elapsed_sec}s with exit_code=${exit_code}"

    if [[ $exit_code -ne 0 ]]; then
      if [[ "$ALLOW_EXIT139_WITH_ARTIFACTS" -eq 1 && "$exit_code" -eq 139 && \
            -f "${run_dir}/CameraTrajectory.txt" && \
            -f "${run_dir}/KeyFrameTrajectory.txt" ]]; then
        echo "  status=success_with_exit139_artifacts_present" >> "${run_dir}/run_meta.txt"
        echo "  note=segfault_after_artifacts; likely teardown/viewer issue" >> "${run_dir}/run_meta.txt"
        echo "  -> run completed with artifacts despite exit_code=139"
        continue
      fi
      if [[ $timed_out -eq 1 ]]; then
        echo "  status=failed (timeout_after=${RUN_TIMEOUT_SEC}s, exit_code=${exit_code})" >> "${run_dir}/run_meta.txt"
      else
        echo "  status=failed (exit_code=${exit_code})" >> "${run_dir}/run_meta.txt"
      fi
      echo "  -> run failed, see ${run_dir}/stderr.log"
    else
      echo "  status=success" >> "${run_dir}/run_meta.txt"
    fi

    for artifact in CameraTrajectory.txt KeyFrameTrajectory.txt; do
      if [[ -f "$ROOT_DIR/$artifact" ]]; then
        mv "$ROOT_DIR/$artifact" "${run_dir}/${artifact}"
      else
        echo "  missing_artifact=${artifact}" >> "${run_dir}/run_meta.txt"
      fi
    done
  done
done

echo
echo "Evaluation finished."
echo "Results are in: $OUTPUT_ROOT"
