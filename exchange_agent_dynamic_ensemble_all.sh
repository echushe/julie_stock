#!/bin/bash

CONFIG="stock_exchange_agent_configs/ALL_60vs08_LSTM064_personal.yaml"
NUM_REPEATS=10
MODEL_POOL_DIR="stock_exchange_agent_records/ALL_60vs08_064/ALL_60vs08_LSTM064_run_model_pool_to_grow/"
LAST_DATE="2025-05-09"

WEEK_DAY=4
ENSEMBLE_UPDATE_INTERVAL=5

# Parameter grids
#TOP_PERCENTAGES=(0.001 0.002 0.004 0.008 0.016 0.032 0.064 0.128 0.256)
#HALF_LIFES=(8 16 32 64)

TOP_PERCENTAGES=(0.064)
HALF_LIFES=(8)

for top in "${TOP_PERCENTAGES[@]}"; do
  for half in "${HALF_LIFES[@]}"; do
    echo "Running: top=${top}, half_life=${half}"
    python exchange_agent_dynamic_ensemble.py \
      --config "$CONFIG" \
      --num_repeats "$NUM_REPEATS" \
      --model_pool_log_dir "$MODEL_POOL_DIR" \
      --last_history_date "$LAST_DATE" \
      --ensemble_update_weekday "$WEEK_DAY" \
      --ensemble_update_interval "$ENSEMBLE_UPDATE_INTERVAL" \
      --top_percentage "$top" \
      --daily_return_half_life "$half"
  done
done