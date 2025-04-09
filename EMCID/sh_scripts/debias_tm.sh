#!/bin/bash

# Ensure that single and multiple experiments do not run at the same time
export PYTHONPATH=.

# Set default values for variables
GPU_RANK=${GPU_RANK:-0}  # Default GPU rank
MOM2=${MOM2:-4000}       # Default momentum parameter
HPARAM=${HPARAM:-"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01"}  # Default hyperparameters
EVAL_COCO=${EVAL_COCO:-0}  # Default evaluation flag for COCO dataset

# Check if COCO evaluation is enabled
COCO_FLAG=""
if [[ ${EVAL_COCO} -eq 1 ]]; then
    COCO_FLAG="--coco"  # Set COCO evaluation flag
fi

# Get the current date and time for logging purposes
current_time=$(date +"%Y-%m-%d_%H-%M-%S")

# Create a directory for logs with the current date and time
log_dir="log/$current_time"
mkdir -p "$log_dir"  # Create the directory if it doesn't exist

# Check the command-line argument to determine which experiment to run
if [ "$1" = "single" ]; then
    nohup python \
    scripts/eval_debias.py \
    --mom2_weight=${MOM2} \
    --device=cuda:2 \
    --hparam=${HPARAM} \
    --seed_num=10 \
    --recompute_factors \
    --max_iters=10 \
    > "$log_dir/debias_single.out" 2>&1 &  # Redirect output to the log file

    echo "log file: $log_dir/debias_single.out"

elif [ "$1" = "multiple" ]; then
    nohup python \
    scripts/eval_debias.py \
    --mom2_weight=4000 \
    --device=cuda:2 \
    --hparam=dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01 \
    --seed_num=10 \
    --recompute_factors \
    --max_iters=10 \
    ${COCO_FLAG} \
    --mixed \
    > "$log_dir/debias_multiple.out" 2>&1 &  # Redirect output to the log file

    echo "log file: $log_dir/debias_multiple.out"

else
    echo "Please input 'single' or 'multiple' to determine which experiment to run."  # Error message for invalid input
fi