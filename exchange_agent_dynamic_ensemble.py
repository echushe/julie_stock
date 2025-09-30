from exchange_agent_ensemble_search import search_ensemble_by_logs
from train_daily_data.model_selection import get_models_via_paths
from train_daily_data.model_cluster import ModelCluster, REGModelCluster, CLSModelCluster
from train_daily_data.global_logger import configure_logger, resume_logger, print_log
from exchange_agent import StockExchangeAgent
from exchange_agent_lb import *

import argparse
import yaml
import torch
import random
import numpy as np
import os
import datetime
import json

def ensemble_dynamic_simulation(args, config):

    config_file_name = os.path.basename(args.config).replace('.yaml', '')
    log_name = config_file_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    configure_logger(log_name, config, True)

    infer_dataset = load_infer_dataset(config, final_test=args.final_test)
    config['my_name'] = config_file_name

    if args.gpu_id >= 0:
        config['device']['gpu_id'] = args.gpu_id
        print_log(f"Overriding GPU id to {args.gpu_id}", level='INFO')

    if args.num_future_days > 0:
        config['inference']['num_future_days'] = args.num_future_days
    if args.top_percentage > 0:
        config['inference']['top_percentage'] = args.top_percentage
    if args.max_ensemble_size > 0:
        config['inference']['max_ensemble_size'] = args.max_ensemble_size

    print_log(json.dumps(config, indent=4), level='INFO')

    # Store ensemble model paths for each future day
    # (In this simulation golden models are cached for repeats)
    ensemble_paths_future_days_as_key = dict()
    for i in range(args.num_future_days + 1):
        if i % args.model_interval == 0:
            last_history_date_l, last_future_date_l, ensemble_paths = \
                search_ensemble_by_logs(
                    args.model_pool_log_dir,
                    config,
                    config['inference']['num_future_days'] - i,
                    config['inference']['top_percentage'],
                    config['inference']['max_ensemble_size'],
                    logging=False)
            ensemble_paths_future_days_as_key[args.num_future_days - i] = (last_history_date_l, ensemble_paths)
            if i == 0:
                last_history_date = last_history_date_l
                last_future_date = last_future_date_l

    final_total_amounts = []

    if args.resume_from_log_dir != '':

        rightful_log_paths, log_date_list = StockExchangeAgent.find_rightful_log_paths(args.resume_from_log_dir)
        log_date_indices = {date: idx for idx, date in enumerate(log_date_list)}

        for log_path in rightful_log_paths:
            resume_logger(log_path, config)

            print_log('###########################################################################################', level='INFO')

            stock_agent = None
            total_amount_record = None
            next_date = None
            distance = -1

            for i in range(args.num_future_days + 1):

                # Redo the search of golden models after each interval 
                # (In this simulation golden models are cached for repeats)
                if i % args.model_interval == 0:
                    last_history_date_l, ensemble_paths = ensemble_paths_future_days_as_key[args.num_future_days - i]
                    print_log(f"Switching models at day {i}, date: {last_history_date_l}, number of future days: {args.num_future_days - i}", level='INFO')

                    if stock_agent is None:
                        # calculate distance (number of trade days) between last_history_date_l and the last date in the log
                        # if distance is smaller than the interval, then we can start resuming the stock agent

                        if last_history_date_l not in log_date_indices:
                            error_msg = f"The last trade date in the log directory {log_date_list[-1]} is likely later than the last history date {last_history_date_l}."
                            error_msg += "You cannot resume from a later date than the last history date."
                            error_msg += "Running tests on validation set leaks future information. It is not allowed."
                            raise ValueError(error_msg)

                        # get index of last_future_date_l in log_date_list
                        last_history_date_index = log_date_indices[last_history_date_l]
                        distance = len(log_date_list) - 1 - last_history_date_index
                        
                        # if distance is smaller than the interval or we have reached the last available model ensemble
                        # it means last_history_date_l is the latest ensemble date before the last date in the log
                        # so we can now initialize the stock agent
                        if distance < args.model_interval or i == args.num_future_days:
                            # Setup stock agent
                            stock_agent = StockExchangeAgent(config, infer_dataset, ModelCluster(None, config), False)
                            models = get_models_via_paths(ensemble_paths, config)
                            if config['model']['autoregressive']:
                                model_cluster = REGModelCluster(models, config)
                            else:
                                model_cluster = CLSModelCluster(models, config)
                            stock_agent.set_model_cluster(model_cluster)
                            next_date = stock_agent.resume_from_log(log_path, last_history_date_l)

                    else:
                        assert last_history_date_l == stock_agent.stock_reservoir.trade_date
                    
                        models = get_models_via_paths(ensemble_paths, config)
                        if config['model']['autoregressive']:
                            model_cluster = REGModelCluster(models, config)
                        else:
                            model_cluster = CLSModelCluster(models, config)
                        stock_agent.set_model_cluster(model_cluster)

                # Consume the distance to reach the last date in the log
                if stock_agent is not None:
                    if distance >= 0:
                        distance -= 1
                    else:
                        if stock_agent.stock_reservoir.trade_date is None:
                            break
                        next_date, total_amount_record = stock_agent.step()
                        if next_date is None:
                            break

            # Continute to finish the inference until the last available trade date
            while next_date is not None:
                next_date, total_amount_record = stock_agent.step()

            if total_amount_record is not None:
                final_total_amounts.append(total_amount_record[-1])

            print_log('###########################################################################################', level='INFO')

    else:

        for i in range(args.num_repeats):
            configure_logger(log_name, config, True)

            print_log('###########################################################################################', level='INFO')

            # Setup stock agent
            stock_agent = StockExchangeAgent(config, infer_dataset, ModelCluster(None, config), False)
            total_amount_record = None

            #for key, value in stock_agent.stock_reservoir.next_trade_date_cache.items():
            #    print(f"Next trade date for {key}: {value}")

            # The date to start should be the last history date
            stock_agent.stock_reservoir.set_trade_date(last_history_date)

            for i in range(args.num_future_days + 1):

                # Redo the search of golden models after each interval 
                # (In this simulation golden models are cached for repeats)
                if i % args.model_interval == 0:
                    last_history_date_l, ensemble_paths = ensemble_paths_future_days_as_key[args.num_future_days - i]
                    print_log(f"Switching models at day {i}, date: {last_history_date_l}, number of future days: {args.num_future_days - i}", level='INFO')
                    assert last_history_date_l == stock_agent.stock_reservoir.trade_date

                    models = get_models_via_paths(ensemble_paths, config)
                    if config['model']['autoregressive']:
                        model_cluster = REGModelCluster(models, config)
                    else:
                        model_cluster = CLSModelCluster(models, config)
                    stock_agent.set_model_cluster(model_cluster)

                next_date, total_amount_record = stock_agent.step()
                if next_date is None:
                    break

            # Continute to finish the inference until the last available trade date
            while next_date is not None:
                next_date, total_amount_record = stock_agent.step()

            if total_amount_record is not None:
                final_total_amounts.append(total_amount_record[-1])

            print_log('###########################################################################################', level='INFO')

    if len(final_total_amounts) == 0:
        return
    configure_logger(log_name, config, True)
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

    parser.add_argument(
            '-ft', '--final_test',
            help='specify to use final test dataset',
            action='store_true'
        )
    
    # dir of logs
    parser.add_argument(
            '-lpld', '--model_pool_log_dir',
            type=str,
            help='directory of logs',
            default='logs'
        )
    
    # Specify number of future days for test
    parser.add_argument(
            '-td', '--num_future_days',
            type=int,
            help='number of future days for test',
            default=-1
        )
    
    # Days interval to changes models
    parser.add_argument(
            '-mi', '--model_interval',
            type=int,
            help='number of days to change models',
            default=-1
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

    ensemble_dynamic_simulation(args, config)