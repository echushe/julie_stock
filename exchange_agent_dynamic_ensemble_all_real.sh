#!/bin/bash

CONFIG="stock_exchange_agent_configs/ALL_60vs08_LSTM064_personal.yaml"
MODEL_POOL_DIR="stock_exchange_agent_records/ALL_60vs08_064/ALL_60vs08_LSTM064_run_model_pool_to_grow/"
RESUME_DIR="stock_exchange_agent_records/ALL_60vs08_064/ALL_60vs08_LSTM064_personal_to_grow/"
LAST_DATE="2025-10-31"

python exchange_agent_dynamic_ensemble.py \
  --config "$CONFIG" \
  --model_pool_log_dir "$MODEL_POOL_DIR" \
  --resume_from_log_dir "$RESUME_DIR" \
  --last_history_date "$LAST_DATE" \
  --final_test
