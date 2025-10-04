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
    all_trade_dates = infer_dataset.get_all_trade_dates()
    all_trade_dates_indices = {date: idx for idx, date in enumerate(all_trade_dates)}

    config['my_name'] = config_file_name

    if args.gpu_id >= 0:
        config['device']['gpu_id'] = args.gpu_id
        print_log(f"Overriding GPU id to {args.gpu_id}", level='INFO')

    # check last_history_date with all_trade_dates
    if args.last_history_date is not None and args.last_history_date != '':
        if args.last_history_date not in all_trade_dates:
            raise ValueError(f"Last history date {args.last_history_date} not in the trade dates of the dataset.")
        last_history_index = all_trade_dates_indices[args.last_history_date]
        num_future_days = len(all_trade_dates) - last_history_index - 1
        if num_future_days <= 0:
            raise ValueError(f"Last history date {args.last_history_date} is the last trade date in the dataset. No future days to test.")
        print_log(f"Setting number of future days to {num_future_days} based on last history date {args.last_history_date}.", level='INFO')
        config["inference"]["last_history_date"] = args.last_history_date

    if args.top_percentage > 0:
        config['inference']['top_percentage'] = args.top_percentage
    if args.max_ensemble_size > 0:
        config['inference']['max_ensemble_size'] = args.max_ensemble_size
    if args.ensemble_update_weekday >= 0 and args.ensemble_update_weekday <= 6:
        config['inference']['ensemble_update_weekday'] = args.ensemble_update_weekday
    if args.ensemble_update_interval > 0:
        config['inference']['ensemble_update_interval'] = args.ensemble_update_interval
    

    print_log(json.dumps(config, indent=4), level='INFO')

    # index of last history date
    last_history_index = all_trade_dates_indices[config["inference"]["last_history_date"]]
    # Dates for test includes the last history date and all future dates
    dates_for_test = all_trade_dates[last_history_index : ]
    dates_for_test_indices = {date: idx for idx, date in enumerate(dates_for_test)}

    # Store ensemble model paths for each future day
    # (In this simulation golden models are cached for repeats)
    ensemble_paths_date_as_key = dict()

    ensemble_update_weekday = config['inference']['ensemble_update_weekday']
    if ensemble_update_weekday < -1 or ensemble_update_weekday > 6:
        raise ValueError("Ensemble update weekday must be between 0 and 6, or -1 if not specified.")
    if ensemble_update_weekday in {5, 6}:
        print_log("Warning: Ensemble update weekday is set to weekend (5=Saturday, 6=Sunday). Changing to Friday (4).", level='WARNING')
        ensemble_update_weekday = 4

    ensemble_update_interval = config['inference']['ensemble_update_interval']
    if ensemble_update_interval <= 0:
        raise ValueError("Ensemble update interval must be a positive integer.")

    if ensemble_update_weekday < 0:
        for i, trade_date in enumerate(dates_for_test):
            if i % ensemble_update_interval == 0:
                last_future_date_l, ensemble_paths = \
                    search_ensemble_by_logs(
                        args.model_pool_log_dir,
                        config,
                        trade_date,
                        config['inference']['top_percentage'],
                        config['inference']['max_ensemble_size'],
                        logging=False)
                ensemble_paths_date_as_key[trade_date] = ensemble_paths
    else:
        for i, trade_date in enumerate(dates_for_test):
            trade_date_obj = datetime.datetime.strptime(trade_date, '%Y-%m-%d')
            weekday = trade_date_obj.weekday()
            if weekday == ensemble_update_weekday:
                last_future_date_l, ensemble_paths = \
                    search_ensemble_by_logs(
                        args.model_pool_log_dir,
                        config,
                        trade_date,
                        config['inference']['top_percentage'],
                        config['inference']['max_ensemble_size'],
                        logging=False)
                ensemble_paths_date_as_key[trade_date] = ensemble_paths

    final_total_amounts = []

    if args.resume_from_log_dir != '':

        rightful_log_paths, log_date_list = StockExchangeAgent.find_rightful_log_paths(args.resume_from_log_dir)
        log_date_indices = {date: idx for idx, date in enumerate(log_date_list)}

        for log_path in rightful_log_paths:
            resume_logger(log_path, config)

            print_log('###########################################################################################', level='INFO')
            print_log(f"Resuming from log path: {log_path}", level='INFO')
            total_amount_record = None
            next_date = None

            trade_dates_and_ensemble_paths = sorted(ensemble_paths_date_as_key.items(), key=lambda x: x[0])
            ensemble_update_dates_until_last_log_date = []

            for i, (trade_date, ensemble_paths) in enumerate(trade_dates_and_ensemble_paths):
                if trade_date > log_date_list[-1]:
                    break
                ensemble_update_dates_until_last_log_date.append(trade_date)

            if len(ensemble_update_dates_until_last_log_date) == 0:
                error_msg = f"No ensemble update date is earlier than or equal to the last trade date in the log directory {log_date_list[-1]}."
                error_msg += "You cannot resume from an earlier date than the ensemble update date (last history date)."
                raise ValueError(error_msg)
            
            stock_agent = StockExchangeAgent(config, infer_dataset, ModelCluster(None, config), False)
            ensemble_paths = ensemble_paths_date_as_key[ensemble_update_dates_until_last_log_date[-1]]
            print_log(f"Switching models, date: {ensemble_update_dates_until_last_log_date[-1]}", level='INFO')
            models = get_models_via_paths(ensemble_paths, config)
            if config['model']['autoregressive']:
                model_cluster = REGModelCluster(models, config)
            else:
                model_cluster = CLSModelCluster(models, config)
            stock_agent.set_model_cluster(model_cluster)
            next_date = stock_agent.resume_from_log(log_path, ensemble_update_dates_until_last_log_date[-1])
            if next_date is None:
                continue

            # find index of log_date_list[-1] in dates_for_test
            if log_date_list[-1] not in dates_for_test_indices:
                error_msg = f"The last trade date in the log directory {log_date_list[-1]} is not within the test dates."
                raise ValueError(error_msg)
            
            last_log_date_index_in_test = dates_for_test_indices[log_date_list[-1]]

            for i in range(last_log_date_index_in_test + 1, len(dates_for_test)):

                trade_date = dates_for_test[i]

                # Redo the search of ensemble models at each update date
                # (In this simulation ensemble models are cached for repeats)
                if trade_date in ensemble_paths_date_as_key:
                    ensemble_paths = ensemble_paths_date_as_key[trade_date]
                    print_log(f"Switching models, date: {trade_date}", level='INFO')
                    assert trade_date == stock_agent.stock_reservoir.trade_date

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
            stock_agent.stock_reservoir.set_trade_date(config["inference"]["last_history_date"])

            for i, trade_date in enumerate(dates_for_test):

                # Redo the search of golden models after each interval 
                # (In this simulation golden models are cached for repeats)
                if trade_date in ensemble_paths_date_as_key:
                    ensemble_paths = ensemble_paths_date_as_key[trade_date]
                    print_log(f"Switching models, date: {trade_date}", level='INFO')
                    assert trade_date == stock_agent.stock_reservoir.trade_date

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
    
    # Specify last history date for test
    parser.add_argument(
            '-td', '--last_history_date',
            type=str,
            help='last history date for test',
            default=''
        )
    
    # Days interval to changes models, if weekday is not specified
    # e.g., if specified 5, then change models every 5 days
    parser.add_argument(
            '-mi', '--ensemble_update_interval',
            type=int,
            help='number of days to change models',
            default=5
        )
    
    # specify weekday to change ensemble models
    # e.g., if specified 0, then change models every Monday
    parser.add_argument(
        '-mcw', '--ensemble_update_weekday',
        type=int,
        help='weekday to change models (0=Monday, 4=Friday)',
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