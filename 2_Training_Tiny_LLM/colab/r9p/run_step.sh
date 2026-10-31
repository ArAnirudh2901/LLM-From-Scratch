#!/usr/bin/env bash
# Runs one r9p pipeline step on Kaggle and keeps only its public results in the repo.
#   colab/r9p/run_step.sh A        (then B1, C, B2, D, E; run from the repository root)
# Full outputs, which hold real question text, stay outside the repo in $KAGGLE_RESULTS (default ~/kaggle_r9p).
set -euo pipefail
STEP="${1:?usage: colab/r9p/run_step.sh A|B1|C|B2|D|E}"
PIPE="${PIPE:-r9p}"   # r10: PIPE=r10 colab/r9p/run_step.sh C
DIR=$(ls -d colab/"$PIPE"/"${STEP}"_* 2>/dev/null | head -1 || true)
[ -n "$DIR" ] || { echo "STOP: no folder colab/$PIPE/${STEP}_* (run from the repository root)"; exit 1; }
ID=$(python3 -c "import json; print(json.load(open('$DIR/kernel-metadata.json'))['id'])")
WORK="${KAGGLE_RESULTS:-$HOME/kaggle_$PIPE}/$STEP"
rm -rf "$WORK" && mkdir -p "$WORK"
if [ "${WATCH:-}" = 1 ]; then   # re-attach to a run already pushed
  echo "watching $ID (no push)"
else
  echo "pushing $DIR as $ID"
  kaggle kernels push -p "$DIR" | tee /dev/stderr | grep -q 'successfully pushed' \
    || { echo "STOP: push of step $STEP failed (Kaggle allows 2 GPU runs at once)"; exit 1; }
  sleep 60
fi
while true; do
  S=$(kaggle kernels status "$ID" 2>&1 | tr '[:upper:]' '[:lower:]' || true)
  echo "$(date +%H:%M) $S"
  case "$S" in
    *complete*) break ;;
    *error*|*cancel*) kaggle kernels output "$ID" -p "$WORK" >/dev/null 2>&1 || true
                      grep -ho 'STOP: [^"\\]*' "$WORK"/*.log 2>/dev/null | head -3
                      echo "STOP: Kaggle run of step $STEP ended with: $S (full log in $WORK)"; exit 1 ;;
  esac
  sleep 600
done
kaggle kernels output "$ID" -p "$WORK"
LOG=$(ls "$WORK"/*.log 2>/dev/null | head -1 || true)
if [ -n "$LOG" ] && grep -q 'STOP: ' "$LOG"; then
  grep -ho 'STOP: [^"\\]*' "$LOG" | head -3; echo "STOP: step $STEP stopped (full log in $LOG)"; exit 1
fi
if [ -z "$LOG" ] || ! grep -q "${STEP:0:1} DONE" "$LOG"; then
  echo "STOP: step $STEP finished without its '${STEP:0:1} DONE' line (full log in $WORK)"; exit 1
fi
mkdir -p "results/$PIPE/$STEP"
cp "$WORK"/public/*.json "results/$PIPE/$STEP/"
echo "step $STEP passed; public results:"; ls "results/$PIPE/$STEP"
SUMMARY=$(ls "results/$PIPE/$STEP"/*summary*.json "results/$PIPE/$STEP"/decision.json 2>/dev/null | head -1 || true)
[ -n "$SUMMARY" ] && cat "$SUMMARY"
