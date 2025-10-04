from exchange_agent import StockExchangeAgent
from train_daily_data.model_selection import get_models_via_checkpoint_dirs, get_models_via_single_model_path, get_models_via_paths
from train_daily_data.model_cluster import ModelCluster, REGModelCluster, CLSModelCluster
from exchange_agent_ensemble_search import search_ensemble_by_logs
from exchange_agent_lb import *
from train_daily_data.global_logger import configure_logger, resume_logger, print_log

import argparse
import yaml
import torch
import random
import numpy as np
import json
import os
import datetime


def ensemble_static_simulation(args, config):

    config_file_name = os.path.basename(args.config).replace('.yaml', '')
    log_name = config_file_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    configure_logger(log_name, config, not args.real_stock_exchange)

    infer_dataset = load_infer_dataset(config, final_test=args.final_test)
    all_trade_dates = infer_dataset.get_all_trade_dates()
    all_trade_dates_indices = {date: idx for idx, date in enumerate(all_trade_dates)}

    config['my_name'] = config_file_name

    if args.gpu_id >= 0:
        config['device']['gpu_id'] = args.gpu_id
        print_log(f"Overriding GPU id to {args.gpu_id}", level='INFO')

    # check last_history_date with all_trade_dates
    if args.last_history_date is not None and args.last_history_date != '':
        last_history_date = args.last_history_date
        if last_history_date not in all_trade_dates:
            raise ValueError(f"Last history date {last_history_date} not in the trade dates of the dataset.")
        last_history_index = all_trade_dates_indices[last_history_date]
        num_future_days = len(all_trade_dates) - last_history_index - 1
        if num_future_days <= 0:
            raise ValueError(f"Last history date {last_history_date} is the last trade date in the dataset. No future days to test.")
        print_log(f"Setting number of future days to {num_future_days} based on last history date {last_history_date}.", level='INFO')
        config["inference"]["last_history_date"] = last_history_date

    if args.top_percentage > 0:
        config["inference"]["top_percentage"] = args.top_percentage
    if args.max_ensemble_size > 0:
        config["inference"]["max_ensemble_size"] = args.max_ensemble_size

    print_log(json.dumps(config, indent=4), level='INFO')

    if args.dummy:
    # If dummy mode is enabled, use dummy prediction instead of real model prediction
        print_log("Using dummy prediction instead of real model prediction.", level='INFO')
        models = None
        model_cluster = ModelCluster(models, config)
    
    else:
        last_future_date_l, ensemble_paths = \
            search_ensemble_by_logs(
                args.model_pool_log_dir,
                config,
                config["inference"]["last_history_date"],
                config['inference']['top_percentage'],
                config['inference']['max_ensemble_size'],
                logging=False)
        
        # The trade date starts with should be the last history date
        # We should not run a newer ensemble for earlier dates
        print_log(f"Last history date: {last_history_date}", level='INFO')
        models = get_models_via_paths(ensemble_paths, config)

        if config['model']['autoregressive']:
            model_cluster = REGModelCluster(models, config)
        else:
            model_cluster = CLSModelCluster(models, config)

    if args.real_stock_exchange:
        configure_logger(log_name, config, not args.real_stock_exchange)

        stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, args.real_stock_exchange)
        stock_agent.run()

    else:
        final_total_amounts = []

        if args.resume_from_log_dir != '':

            rightful_log_paths, date_list = StockExchangeAgent.find_rightful_log_paths(args.resume_from_log_dir)

            if date_list is not None and len(date_list) > 0 and date_list[-1] < last_history_date:
                error_msg = f"The last trade date in the log directory {date_list[-1]} is earlier than the last history date {last_history_date}."
                error_msg += "You cannot resume from an earlier date than the last history date."
                error_msg += "Running tests on validation set leaks future information. It is not allowed."
                raise ValueError(error_msg)

            for log_path in rightful_log_paths:
                resume_logger(log_path, config)

                print_log('###########################################################################################', level='INFO')
        
                print_log(f"Resuming from log path: {log_path}", level='INFO')
                stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, False)
                stock_agent.resume_from_log(log_path, last_history_date)
                final_total_amounts.append(stock_agent.run()[-1])

                print_log('###########################################################################################', level='INFO')
        
        else:

            for i in range(args.num_repeats):
                configure_logger(log_name, config, not args.real_stock_exchange)

                print_log('###########################################################################################', level='INFO')

                stock_agent = StockExchangeAgent(config, infer_dataset, model_cluster, args.real_stock_exchange)
                if not args.real_stock_exchange and last_history_date is not None:
                    stock_agent.stock_reservoir.set_trade_date(last_history_date)
                
                final_total_amounts.append(stock_agent.run()[-1])

                print_log('###########################################################################################', level='INFO')

        configure_logger(log_name, config, not args.real_stock_exchange)
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
    
    # Real stock exchange mode or simulation mode
    parser.add_argument(
            '-rse', '--real_stock_exchange',
            help='use real stock exchange mode or simulation mode',
            action='store_true'
        )

    parser.add_argument(
            '-ft', '--final_test',
            help='specify to use final test dataset',
            action='store_true'
        )
    
    # dir of model pool logs for golden model search
    parser.add_argument(
            '-lpld', '--model_pool_log_dir',
            type=str,
            help='directory of model_pool logs',
            default='logs'
        )

    # Specify last history date for test
    parser.add_argument(
            '-td', '--last_history_date',
            type=str,
            help='last history date for test',
            default=''
        )
    
    # top percentage
    parser.add_argument(
        '-tp', '--top_percentage',
        type=float,
        help='top percentage of models to consider',
        default=-0.1
        )

    # Specify max size of ensemble
    parser.add_argument(
            '-me', '--max_ensemble_size',
            type=int,
            help='maximum size of ensemble',
            default=-1
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

    ensemble_static_simulation(args, config)