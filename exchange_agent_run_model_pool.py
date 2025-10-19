
from train_daily_data.model_selection import \
    resolve_gpu_plan, get_all_model_paths_via_checkpoint_dirs, get_models_via_single_model_path
from train_daily_data.model_cluster import ModelCluster, REGModelCluster, CLSModelCluster
from exchange_agent_lb import *
from train_daily_data.global_logger import configure_logger, print_log, resume_logger
from exchange_agent import StockExchangeAgent
from exchange_agent_evaluate import evaluate_equity_curve

import argparse
import yaml
import torch
import random
import numpy as np
import os
import datetime
import json
import copy
import multiprocessing
import time
import re


def inference_main_of_one_model(
        model_path,
        model_idx,
        gpu_id,
        config,
        log_name,
        num_repeats,
        infer_dataset,
        model_idx_path_and_scores,
        # Optional argument to resume from a specific log directory
        # If specified, the agent will resume from the last trade date in that log directory
        dir_to_resume_from=None
    ):

    sub_log_name = log_name + f'_model{model_idx:03d}'
    configure_logger(sub_log_name, config, log_to_file=True)
    print_log(f"Running single model from {model_path}", level='INFO')

    model = get_models_via_single_model_path(model_path, config, cpu_mode=True)[model_path]

    try:
        # Deep copy the model from the mother process to avoid modifying the original and ensure multi-GPU compatibility
        #model = copy.deepcopy(model)
        model = model.to('cuda:{}'.format(gpu_id))
    except RuntimeError as e:
        print_log(f"RuntimeError occurred while moving model to GPU: {e}", level='ERROR')
        return
    except OverflowError as e:
        print_log(f"OverflowError occurred while moving model to GPU: {e}", level='ERROR')
        return
    except MemoryError as e:
        print_log(f"MemoryError occurred while moving model to GPU: {e}", level='ERROR')
        return
    except Exception as e:
        print_log(f"Error occurred while moving model to GPU: {e}", level='ERROR')
        return

    print_log(f"Single model {model_idx} created successfully", level='INFO')

    time.sleep(1)  # Sleep for a second to avoid log overlap

    if config['model']['autoregressive']:
        model_cluster = REGModelCluster({model_path:model}, config)
    else:
        model_cluster = CLSModelCluster({model_path:model}, config)

    total_amount_record_2d = []

    if dir_to_resume_from is not None:
        rightful_log_paths, date_list = StockExchangeAgent.find_rightful_log_paths(dir_to_resume_from)

        for log_path in rightful_log_paths:
            resume_logger(log_path, config)
            print_log('###########################################################################################', level='INFO')
    
            print_log(f"Resuming from log path: {log_path}", level='INFO')
            try:
                stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, False)
                stock_agent.resume_from_log(log_path)
                total_amount_record_2d.append(stock_agent.run())
            except Exception as e:
                print_log(f"Error occurred while resuming stock agent: {e}", level='ERROR')
            print_log('###########################################################################################', level='INFO')

    else:
        for i in range(num_repeats):
            configure_logger(sub_log_name, config, log_to_file=True)

            print_log('###########################################################################################', level='INFO')

            try:
                stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, real_stock_exchange_mode=False)
                total_amount_record_2d.append(stock_agent.run())
            except Exception as e:
                print_log(f"Error occurred while running stock agent: {e}", level='ERROR')

            print_log('###########################################################################################', level='INFO')
    

    configure_logger(sub_log_name, config, log_to_file=True)
    if len(total_amount_record_2d) == 0:
        print_log(f"No valid total amounts for model {model_path}.", level='ERROR')
        return

    decay_50day = config["ensemble_search"]["daily_return_50day_decay"]
    volatility_rate = config["ensemble_search"]["volatility_rate"]
    drawdown_rate = config["ensemble_search"]["drawdown_rate"]
    min_weekly_win_rate = config["ensemble_search"]["min_weekly_win_rate"]
    min_monthly_win_rate = config["ensemble_search"]["min_monthly_win_rate"]
    min_weekly_IR_median = config["ensemble_search"]["min_weekly_IR_median"]
    min_monthly_IR_median = config["ensemble_search"]["min_monthly_IR_median"]

    score, _ = evaluate_equity_curve(
        total_amount_record_2d,
        decay_50day,
        volatility_rate,
        drawdown_rate,
        min_weekly_win_rate,
        min_monthly_win_rate,
        min_weekly_IR_median,
        min_monthly_IR_median
    )

    model_idx_path_and_scores.append((model_idx, model_path, score))


def model_subdirs_resume_from(log_dir):

    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} does not exist.")

    # get all dirs ending with model[0-9]+ under log_dir (just under log_dir, no recursion)
    model_log_dirs = [d for d in os.listdir(log_dir) if re.match(r'.*_model[0-9]+', d)]
    model_log_dirs = sorted(model_log_dirs)

    model_log_dirs_model_path_as_key = dict()
    for model_log_dir in model_log_dirs:
        print_log(f"Processing log directory: {os.path.join(log_dir, model_log_dir)}", level='INFO')
        # get all log files under model_log_dir
        log_files = [f for f in os.listdir(os.path.join(log_dir, model_log_dir)) if f.endswith('.log')]
        log_files = sorted(log_files)
        if len(log_files) > 1:
            # get model path from the first log file
            first_log_file_path = os.path.join(log_dir, model_log_dir, log_files[0])
            with open(first_log_file_path, 'r') as f:
                lines = f.readlines()
                model_path = lines[0].strip().split(' ')[-1]
                model_log_dirs_model_path_as_key[model_path] = os.path.join(log_dir, model_log_dir)

    return model_log_dirs_model_path_as_key


def stock_exchange_agent_run_model_pool(args, config):

    config_file_name = os.path.basename(args.config).replace('.yaml', '')

    now = datetime.datetime.now()
    now_as_str = now.strftime('%Y-%m-%d_%H-%M-%S')
    now_as_str_date = now.strftime('%Y-%m-%d')

    log_name = config_file_name + f'_run_model_pool_{now_as_str}/' + now_as_str

    configure_logger(log_name, config, log_to_file=True)
    print_log(json.dumps(config, indent=4), level='INFO')

    infer_dataset = load_infer_dataset(config, final_test=args.final_test)
    config['my_name'] = config_file_name

    if args.gpu_id >= 0:
        config['device']['gpu_id'] = args.gpu_id

    print_log(json.dumps(config, indent=4), level='INFO')

    model_log_dirs_model_path_as_key = dict()

    if args.resume_from_log_dir != '':
        print_log(f"Resuming from log directory: {args.resume_from_log_dir}", level='INFO')
        model_log_dirs_model_path_as_key = model_subdirs_resume_from(args.resume_from_log_dir)
        if len(model_log_dirs_model_path_as_key) == 0:
            raise ValueError(f"No valid model log directories found in {args.resume_from_log_dir}.")
        print_log(f"Found {len(model_log_dirs_model_path_as_key)} models to resume from.", level='INFO')

    else:
        checkpoint_dirs = config['model_pool']['checkpoint_dirs']

        # Get all model paths from the checkpoint directories
        model_paths = get_all_model_paths_via_checkpoint_dirs(checkpoint_dirs, config)
        print_log(f"Found {len(model_paths)} models in checkpoint directories.", level='INFO')
        for model_path in model_paths:
            model_log_dirs_model_path_as_key[model_path] = None  # No log dir since not resuming

    pairs_of_model_path_and_log_dir = sorted(list(model_log_dirs_model_path_as_key.items()), key=lambda x: x[0])

    max_number_of_processes = config['model_pool']['max_number_of_processes']

    multiprocessing.set_start_method('spawn')
    model_idx_path_and_scores = multiprocessing.Manager().list()

    pool_results = []
    pool = None

    # Go through each model and run inference in parallel
    for model_idx in range(len(pairs_of_model_path_and_log_dir)):

        path, model_log_dir = pairs_of_model_path_and_log_dir[model_idx]

        if model_log_dir is not None:
            log_dir_base_name = os.path.basename(model_log_dir)
            model_idx_in_log_dir = int(re.search(r'model([0-9]+)', log_dir_base_name).group(1))
            if model_idx_in_log_dir != model_idx:
                raise ValueError(f"Model index mismatch: {model_idx_in_log_dir} != {model_idx}")

        if len(pool_results) % max_number_of_processes == 0:
            
            if len(pool_results) > 0:
                # Wait for any process to finish
                pool.close()
                pool.join()
                pool_results.clear()

            # Each round only execute up to max_number_of_processes
            n_processes = min(max_number_of_processes, len(pairs_of_model_path_and_log_dir) - model_idx)
            pool = multiprocessing.Pool(n_processes)

            gpu_plan_queue = resolve_gpu_plan(n_models=n_processes, config=config)
            gpu_plan_queue_head = gpu_plan_queue.pop(0)

        while gpu_plan_queue_head[1] == 0:
            # If the head GPU has no more models to load, pop the next one
            if len(gpu_plan_queue) == 0:
                raise ValueError('No more GPUs available in gpu_plan.')
            gpu_plan_queue_head = gpu_plan_queue.pop(0)

        # Decrease the number of models for this GPU
        gpu_id = gpu_plan_queue_head[0]
        gpu_plan_queue_head = (gpu_plan_queue_head[0], gpu_plan_queue_head[1] - 1)

        pool_result = pool.apply_async(inference_main_of_one_model,
                        args=(
                            path,
                            model_idx,
                            gpu_id,
                            config,
                            log_name,
                            args.num_repeats,
                            infer_dataset,
                            model_idx_path_and_scores,
                            model_log_dir
                        ))
        pool_results.append(pool_result)

        print_log(f"Starting process for model {model_idx:03} {path}...", level='INFO')

        model_idx += 1
    
    if len(pool_results) > 0:
        # Wait for any process to finish
        pool.close()
        pool.join()
        pool_results.clear()

    configure_logger(log_name, config, log_to_file=True)

    # Sort the pairs by the final amounts in descending order
    model_idx_path_and_scores = list(model_idx_path_and_scores)
    model_idx_path_and_scores.sort(key=lambda x: x[2], reverse=True)

    # Log the sorted results
    print_log("Sorted model paths by final amounts:", level='INFO')
    for model_idx, model_path, final_amount in model_idx_path_and_scores:
        print_log(f"Model {model_idx} Path: {model_path}, Final amount: {final_amount}", level='INFO')


if __name__ == '__main__': 


    parser = argparse.ArgumentParser(
            description='A sample script demonstrating argparse usage',
            epilog='Example: python script.py -n John --age 25'
        )

    parser.add_argument(
            '-c', '--config',
            type=str,
            help='path of config file',
            default='cls_config.yaml'
        )
    
    # Argument to use constant random seed or not
    parser.add_argument(
            '-ucs', '--use_constant_seed',
            help='specify to use constant random seed',
            action='store_true'
        )
    
    parser.add_argument(
            '-r', '--num_repeats',
            type=int,
            help='number of repeats for the experiment',
            default=1
        )
    
    # Specify GPU id
    # This specification will override the GPU id in the config file
    parser.add_argument(
            '-g', '--gpu_id',
            type=int,
            help='GPU id to use for training',
            default=-1
        )

    parser.add_argument(
            '-ft', '--final_test',
            help='specify to use final test dataset',
            action='store_true'
        )
    
    # resume from existing log directory
    # If specified, the script will check the existing log directory and resume from the last trade date
    parser.add_argument(
            '-rfld', '--resume_from_log_dir',
            type=str,
            help='specify the log directory to resume from',
            default=''
    )

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.use_constant_seed:
        # Set the random seed for reproducibility
        # This is useful for debugging and testing purposes
        # It ensures that the results are consistent across different runs
        # However, it may not be suitable for production use
        # because it can lead to overfitting to the specific random seed used
        # and may not generalize well to other random seeds or real-world data.
        # In production, it's better to use a different random seed each time
        # to ensure that the model is robust and can handle different scenarios.
        # But for this script, we will use a constant random seed for reproducibility.
        print("Using constant random seed for reproducibility.")
        # Set a seed for reproducibility
        seed = 42
        # Set seed for Python's random module
        random.seed(seed)
        # Set seed for NumPy
        np.random.seed(seed)
        # Set seed for PyTorch
        torch.manual_seed(seed)
        # Set seed for CUDA (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    else:
        print("Using variable random seed for each run.")
        # Use a random seed based on OS entropy source
        random.seed(int.from_bytes(os.urandom(4), 'big'))
        # Set seed for NumPy
        np.random.seed(int.from_bytes(os.urandom(4), 'big'))
        # Set seed for PyTorch
        torch.manual_seed(int.from_bytes(os.urandom(4), 'big'))

        if torch.cuda.is_available():
            # Set seed for CUDA (if available)
            torch.cuda.manual_seed(int.from_bytes(os.urandom(4), 'big'))
            # Set seed for all GPUs (if available)
            torch.cuda.manual_seed_all(int.from_bytes(os.urandom(4), 'big'))

    stock_exchange_agent_run_model_pool(args, config)
