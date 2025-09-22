
from train_daily_data.model_selection import get_models_via_checkpoint_dirs
from train_daily_data.model_cluster import ModelCluster, REGModelCluster, CLSModelCluster
from exchange_agent_lb import *
from train_daily_data.global_logger import configure_logger, print_log
from decimal import Decimal, ROUND_HALF_UP, ROUND_UP, getcontext
from exchange_agent import StockExchangeAgent

import argparse
import yaml
import torch
import random
import numpy as np
import os
import datetime
import json


def stock_exchange_agent_balanced_search(args, config, n_models_in_each_checkpoint_dir=10):

    config_file_name = os.path.basename(args.config).replace('.yaml', '')

    log_name = config_file_name \
        + f'_{n_models_in_each_checkpoint_dir:02d}' + f'_repeat{args.num_repeats:03d}_' \
        + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    configure_logger(log_name, config, log_to_file=True)
    print_log(json.dumps(config, indent=4), level='INFO')

    infer_dataset = load_infer_dataset(config, final_test=args.final_test)
    config['my_name'] = config_file_name
    config['model_pool']['n_models_in_each_checkpoint_dir'] = n_models_in_each_checkpoint_dir

    if args.gpu_id >= 0:
        config['device']['gpu_id'] = args.gpu_id

    print_log(json.dumps(config, indent=4), level='INFO')

    if args.dummy:
    # If dummy mode is enabled, use dummy prediction instead of real model prediction
        print_log("Using dummy prediction instead of real model prediction.", level='INFO')
        models = None
        model_cluster = ModelCluster(models, config)
    else:
        checkpoint_dirs = config['model_pool']['checkpoint_dirs']
        models = get_models_via_checkpoint_dirs(checkpoint_dirs, config)

        if config['model']['autoregressive']:
            model_cluster = REGModelCluster(models, config)
        else:
            model_cluster = CLSModelCluster(models, config)

    final_total_amounts = []

    for i in range(args.num_repeats):
        configure_logger(log_name, config, log_to_file=True)

        print_log('###########################################################################################', level='INFO')

        stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, real_stock_exchange_mode=False)
        final_total_amounts.append(stock_agent.run()[-1])

        print_log('###########################################################################################', level='INFO')
        print_log('###########################################################################################', level='INFO')
        print_log('###########################################################################################', level='INFO')


    print_log(f"Mean of final total amounts: {sum(final_total_amounts) / len(final_total_amounts)}", level='INFO')
    themax = max(final_total_amounts)
    themin = min(final_total_amounts)
    print_log(f"Max of final total amounts: {themax}", level='INFO')
    print_log(f"Min of final total amounts: {themin}", level='INFO')

    each_range = (themax - themin) / 10
    thresholds = [themin + i * each_range for i in range(11)]
    
    thresholds[-1] += Decimal(0.01)  # Adjust the last threshold to include the max value
    
    count_ranges = [0] * 10
    for amount in final_total_amounts:
        for i in range(10):
            if thresholds[i] <= amount < thresholds[i + 1]:
                count_ranges[i] += 1
                break
    # Print the count of final total amounts in each range
    print_log("Count of final total amounts in each range:", level='INFO')
    for i in range(10):
        print_log(f"{thresholds[i]:.2f} - {thresholds[i + 1]:.2f}: {count_ranges[i]}", level='INFO')


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
    
    # dummy option to test the script without real models
    parser.add_argument(
            '-d', '--dummy',
            help='use dummy prediction instead of real model prediction',
            action='store_true')
    
    parser.add_argument(
            '-nr', '--n_models_in_each_checkpoint_dir_range',
            type=str,
            help='number range of models in each checkpoint directory',
            default='1-10'
        )
    
    parser.add_argument(
            '-ft', '--final_test',
            help='specify to use final test dataset',
            action='store_true'
        )


    args = parser.parse_args()

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
        # Set a random seed based on current time
        seed_base = int(datetime.datetime.now().timestamp() * 1000) % 2**32 # to ensure it fits in 32 bits
        random.seed(seed_base)

        seed = random.randint(0, 2**32 - 1)
        np.random.seed(seed)

        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            seed = random.randint(0, 2**32 - 1)
            torch.cuda.manual_seed(seed)
            seed = random.randint(0, 2**32 - 1)
            torch.cuda.manual_seed_all(seed)


        # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    range_str = args.n_models_in_each_checkpoint_dir_range
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        n_models_in_each_checkpoint_dir_range = range(start, end + 1)
    else:
        n_models_in_each_checkpoint_dir_range = [int(range_str)]
    for i in n_models_in_each_checkpoint_dir_range:
        stock_exchange_agent_balanced_search(args, config, n_models_in_each_checkpoint_dir=i)

    

    


