#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAIN_BIN="${MAIN_BIN:-$ROOT_DIR/main_openloris}"
BASE_CONFIG="${BASE_CONFIG:-$ROOT_DIR/Examples/config/openloris_input.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/output_data/openloris2}"
RUNS_PER_CONFIG="${RUNS_PER_CONFIG:-5}"
RUN_HEARTBEAT_SEC="${RUN_HEARTBEAT_SEC:-30}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-0}"
ALLOW_EXIT139_WITH_ARTIFACTS="${ALLOW_EXIT139_WITH_ARTIFACTS:-1}"

# Runtime toggles passed through CLI (CLI overrides YAML in main_openloris).
PANGOLIN_VIEWER="${PANGOLIN_VIEWER:-1}"
OBJECT_DETECTION_VIEWER="${OBJECT_DETECTION_VIEWER:-0}"
PACKAGE_PATH="${PACKAGE_PATH:-$ROOT_DIR}"

# Dataset setup. By default use in-repo OpenLORIS package under input_data.
# Override OPENLORIS_PACKAGE_DIR / SETTINGS_FILE / SEQUENCE_BASE_DIR when needed.
OPENLORIS_PACKAGE_DIR="${OPENLORIS_PACKAGE_DIR:-$ROOT_DIR/input_data}"
SEQUENCE_BASE_DIR="${SEQUENCE_BASE_DIR:-$OPENLORIS_PACKAGE_DIR}"
SETTINGS_FILE="${SETTINGS_FILE:-$OPENLORIS_PACKAGE_DIR/openloris.yaml}"

#"market1-1" 
#"market1-2" 
SEQUENCES=("market1-3")

# Format: label|kernel|dynamic_filtering
#"Huber|Huber|0"
#"Barron|Barron|0"
#"BarronSignalHuber|BarronSignalHuber|0"
#"Barron_dynamic_filter|Barron|1"
CONFIGS=(
  "Huber_dynamic_filter|Huber|1"
)

if [[ ! -x "$MAIN_BIN" ]]; then
  echo "Error: executable not found or not executable: $MAIN_BIN"
  echo "Build first (e.g. ./build.sh)."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Warning: BASE_CONFIG not found at $BASE_CONFIG (continuing with generated per-run configs)."
fi
if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "Warning: SETTINGS_FILE not found at $SETTINGS_FILE (continuing; main_openloris may fail if this is required)."
fi

TOTAL_RUNS=$(( ${#SEQUENCES[@]} * ${#CONFIGS[@]} * RUNS_PER_CONFIG ))
CURRENT_RUN=0

echo "Starting OpenLORIS evaluation"
echo "Root: $ROOT_DIR"
echo "Output: $OUTPUT_ROOT"
echo "Total runs: $TOTAL_RUNS"
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

  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r label kernel dynamic_filtering <<< "$cfg"

    for ((run_idx=1; run_idx<=RUNS_PER_CONFIG; run_idx++)); do
      CURRENT_RUN=$((CURRENT_RUN + 1))
      run_dir="${OUTPUT_ROOT}/${sequence}/${label}/run_${run_idx}"
      mkdir -p "$run_dir"

      tmp_config="${run_dir}/openloris_config.yaml"
      cat > "$tmp_config" <<EOF
%YAML:1.0
---
settings_file: "$SETTINGS_FILE"
sequence_path: "$seq_path"
association_file: "$assoc_path"
robust_kernel: "$kernel"
dynamic_filtering: $dynamic_filtering
object_detection_viewer: $OBJECT_DETECTION_VIEWER
EOF

      rm -f "$ROOT_DIR/CameraTrajectory.txt" "$ROOT_DIR/KeyFrameTrajectory.txt" "$ROOT_DIR/alpha_c_log.txt"

      echo "[$CURRENT_RUN/$TOTAL_RUNS] sequence=${sequence}, config=${label}, run=${run_idx}"
      echo "  kernel=$kernel, dynamic_filtering=$dynamic_filtering, object_detection_viewer=$OBJECT_DETECTION_VIEWER" \
        > "${run_dir}/run_meta.txt"
      echo "  config_file=$tmp_config" >> "${run_dir}/run_meta.txt"
      echo "  pangolin_viewer=$PANGOLIN_VIEWER" >> "${run_dir}/run_meta.txt"
      echo "  heartbeat_sec=$RUN_HEARTBEAT_SEC, timeout_sec=$RUN_TIMEOUT_SEC" >> "${run_dir}/run_meta.txt"
      echo "  launching main_openloris..."

      run_start_epoch="$(date +%s)"
      "$MAIN_BIN" \
        "$tmp_config" \
        "$kernel" \
        "$dynamic_filtering" \
        "$PANGOLIN_VIEWER" \
        "$OBJECT_DETECTION_VIEWER" \
        "$PACKAGE_PATH" \
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
        # Some viewer/X11 teardown paths can segfault on process exit after successful tracking.
        # If expected artifacts were produced, treat exit 139 as a non-fatal completed run.
        if [[ "${ALLOW_EXIT139_WITH_ARTIFACTS:-1}" -eq 1 && "$exit_code" -eq 139 && \
              -f "${run_dir}/CameraTrajectory.txt" && \
              -f "${run_dir}/KeyFrameTrajectory.txt" && \
              -f "${run_dir}/alpha_c_log.txt" ]]; then
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

      for artifact in CameraTrajectory.txt KeyFrameTrajectory.txt alpha_c_log.txt; do
        if [[ -f "$ROOT_DIR/$artifact" ]]; then
          mv "$ROOT_DIR/$artifact" "${run_dir}/${artifact}"
        else
          echo "  missing_artifact=${artifact}" >> "${run_dir}/run_meta.txt"
        fi
      done

    done
  done
done

echo
echo "Evaluation finished."
echo "Results are in: $OUTPUT_ROOT"
