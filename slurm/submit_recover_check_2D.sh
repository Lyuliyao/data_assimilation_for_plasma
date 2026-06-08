#!/bin/bash
# Submit the 2D2V recover-check as 3 parallel array tasks (one per
# formulation) + a dependent aggregation job. Prints both job IDs.
#
# Usage:  slurm/submit_recover_check_2D.sh configs/<name>.yaml
set -euo pipefail

CONFIG="${1:?usage: $0 CONFIG.yaml}"
if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

# Array job: 3 tasks, ~10-15 min each (depends on n_steps).
ARRAY_OUT=$(sbatch slurm/recover_check_2D_array.sb "$CONFIG")
ARRAY_JOB=$(echo "$ARRAY_OUT" | awk '{print $NF}')
echo "submitted array job: $ARRAY_JOB"

# Aggregator: waits for the whole array to succeed.
AGG_OUT=$(sbatch --dependency=afterok:"$ARRAY_JOB" slurm/aggregate_recover_check_2D.sb "$CONFIG")
AGG_JOB=$(echo "$AGG_OUT" | awk '{print $NF}')
echo "submitted aggregator: $AGG_JOB (deps on $ARRAY_JOB)"

echo
echo "monitor:   squeue -j ${ARRAY_JOB},${AGG_JOB}"
echo "log roots: logs/mfda-recover-2D-${ARRAY_JOB}_{0..4}.{out,err}"
echo "           logs/mfda-recover-2D-agg-${AGG_JOB}.{out,err}"
