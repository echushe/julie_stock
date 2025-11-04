
from exchange_agent_lb import *
from train_daily_data.global_logger import configure_logger, print_log
from train_daily_data.model_selection import get_all_model_paths_via_checkpoint_dirs
from exchange_agent_evaluate import evaluate_equity_curve, select_distinct_candidates
from visualize_log import primary_process

import argparse
import yaml
import os
import re


class ModelProfile:
    def __init__(self, log_path, score, results, total_amount_list_2d):
        self.log_path = log_path
        self.score = score
        self.results = results
        self.total_amount_list_2d = total_amount_list_2d

class LogDataCache:
    model_pool_log_dir = ''
    data_group = None


def load_data_from_model_pool_logs(model_pool_log_dir, model_paths_available, logging=True):

    # get all sub dirs ending with regular expression model[0-9]+ under model_pool_log_dir
    sub_log_dirs = []
    for root, dirs, files in os.walk(model_pool_log_dir):
        for dir_name in dirs:
            if re.match(r'[0-9\-]+_[0-9\-]+_model[0-9]+', dir_name):

                log_files = [f for f in os.listdir(os.path.join(root, dir_name)) if f.endswith('.log')]
                log_files = sorted(log_files)
                if len(log_files) > 1:
                    # get model path from the first log file
                    first_log_file_path = os.path.join(root, dir_name, log_files[0])
                    with open(first_log_file_path, 'r') as f:
                        lines = f.readlines()
                        model_path = lines[0].strip().split(' ')[-1]
                        if model_paths_available is None or \
                            (model_paths_available is not None and model_path in model_paths_available):
                            sub_log_dirs.append(os.path.join(root, dir_name))

    data_group = []
    for sub_log_dir in sub_log_dirs:
        if logging:
            print_log(f"Processing log directory: {sub_log_dir}", level='INFO')
        result = primary_process(None, sub_log_dir)
        data_group.append(result)
    
    return data_group


def search_ensemble_by_logs(model_pool_log_dir, config, last_history_date, top_percentage, num_to_select, model_paths_available=None, logging=True):

    print_log(f"Searching for model ensemble in logs directory: {model_pool_log_dir}", level='INFO')
    print_log(f"Last history date for test: {last_history_date}", level='INFO')

    if LogDataCache.data_group is None or LogDataCache.model_pool_log_dir != model_pool_log_dir:
        data_group = load_data_from_model_pool_logs(model_pool_log_dir, model_paths_available, logging=logging)
        LogDataCache.data_group = data_group
        LogDataCache.model_pool_log_dir = model_pool_log_dir
    else:
        data_group = LogDataCache.data_group
        if logging:
            print_log(f"Using cached log data with {len(data_group)} entries.", level='INFO')

    print_log(f"Number of log directories (candidate models) processed: {len(data_group)}", level='INFO')

    daily_return_half_life = config["ensemble_search"]["daily_return_half_life"]
    volatility_rate = config["ensemble_search"]["volatility_rate"]
    drawdown_rate = config["ensemble_search"]["drawdown_rate"]
    min_weekly_win_rate = config["ensemble_search"]["min_weekly_win_rate"]
    min_monthly_win_rate = config["ensemble_search"]["min_monthly_win_rate"]
    min_weekly_IR_median = config["ensemble_search"]["min_weekly_IR_median"]
    min_monthly_IR_median = config["ensemble_search"]["min_monthly_IR_median"]

    last_future_date = None

    model_profiles = []
    max_num_of_dates = 0
    for log_path, date_list, total_amount_list_2d in data_group:
        if len(date_list) > max_num_of_dates:
            max_num_of_dates = len(date_list)

    new_data_group = []
    all_trade_dates_in_log = None
    for i, (log_path, date_list, total_amount_list_2d) in enumerate(data_group):
        if len(date_list) < max_num_of_dates:
            print_log(f"Warning: Log {log_path} has only {len(date_list)} days, less than max {max_num_of_dates}. It may affect evaluation.", level='WARNING')
            continue
        if all_trade_dates_in_log is None:
            all_trade_dates_in_log = date_list
        else:
            # compare each date in date_list with all_trade_dates
            if date_list != all_trade_dates_in_log:
                raise ValueError(f"Date list in log {log_path} does not match the common trade dates.")
        new_data_group.append((log_path, date_list, total_amount_list_2d))

    # convert all date objects to strings of format 'YYYY-MM-DD'
    all_trade_dates_in_log = [date.strftime('%Y-%m-%d') for date in all_trade_dates_in_log]
    all_trade_dates_indices = {date: idx for idx, date in enumerate(all_trade_dates_in_log)}

    #for date in all_trade_dates_in_log:
    #    print_log(f"Trade date in logs: {date}", level='INFO')
    if last_history_date not in all_trade_dates_indices:
        raise ValueError(f"Last history date {last_history_date} not found in trade dates from logs.")
    last_history_index = all_trade_dates_indices[last_history_date]
    num_future_days = len(all_trade_dates_in_log) - last_history_index - 1

    data_group = new_data_group

    for log_path, date_list, total_amount_list_2d in data_group:

        #print_log(f'shape of total_amount_list_2d: {total_amount_list_2d.shape}', level='INFO')
        score, results = evaluate_equity_curve(
            total_amount_list_2d,
            daily_return_half_life,
            volatility_rate,
            drawdown_rate,
            min_weekly_win_rate,
            min_monthly_win_rate,
            min_weekly_IR_median,
            min_monthly_IR_median,
            num_future_days,
            logging
        )
        model_profiles.append(ModelProfile(log_path, score, results, total_amount_list_2d))

        if last_future_date is None:
            last_future_date = date_list[-1]

    # sort candidates by score
    model_profiles.sort(key=lambda x: x.score, reverse=True)
    sort_result_str = ''
    for i, profile in enumerate(model_profiles):
        sort_result_str += f"\n{i:>3} Log dir: {os.path.basename(profile.log_path)},"
        #for key, value in profile.results.items():
        #    sort_result_str += f"   {key}: {value:8.4f}"
        sort_result_str += f"   Score: {profile.score:8.4f}"

    if logging:
        print_log(sort_result_str, level='INFO')

    total_amount_list_2d_list = []
    for i, profile in enumerate(model_profiles):
        # ignore negative score candidates
        #if score < 0:
        #    break
        # List all files under log_path
        log_files = os.listdir(profile.log_path)
        log_files = sorted(log_files)
        # get first log file and read it
        if len(log_files) == 0:
            print_log(f"No log files found in {profile.log_path}", level='WARNING')
            continue

        # get first log file and read it
        first_log_file = log_files[0]
        with open(os.path.join(profile.log_path, first_log_file), 'r') as f:
            lines = f.readlines()
            model_path = lines[0].strip().split(' ')[-1]
            profile.model_path = model_path
            total_amount_list_2d_list.append(profile.total_amount_list_2d)

    # pick up top_percentage candidates
    num_candidates = int(len(model_profiles) * top_percentage)
    model_profiles = model_profiles[:num_candidates]
    total_amount_list_2d_list = total_amount_list_2d_list[:num_candidates]

    # remove candidates with negative score
    new_model_profiles = []
    new_total_amount_list_2d_list = []
    for i in range(len(model_profiles)):
        if model_profiles[i].score >= 0:
            new_model_profiles.append(model_profiles[i])
            new_total_amount_list_2d_list.append(total_amount_list_2d_list[i])
    model_profiles = new_model_profiles
    total_amount_list_2d_list = new_total_amount_list_2d_list

    num_to_select = min(num_to_select, len(model_profiles))
    print_log(f'num of top score candidates: {len(model_profiles)}', level='INFO')
    print_log(f'num of least covariance candidates to select: {num_to_select}', level='INFO')

    assert len(model_profiles) > 0

    selected_indices = select_distinct_candidates(total_amount_list_2d_list, num_to_select, num_future_days)
    model_profiles = [model_profiles[idx] for idx in selected_indices]

    print_log(f'least covariance candidates selected: {selected_indices}', level='INFO')
    for i in range(len(model_profiles)):
        idx = selected_indices[i]
        log_path = model_profiles[i].log_path
        print_log(f"Selected golden model {idx:>3}: {re.search(r'model([0-9]+)', log_path).group(1)}", level='INFO')
    print_log("\n", level='INFO')

    if logging:
        ensembles_str = 'Top golden models'
        for log_profile in model_profiles:
            ensembles_str += f"\n    \'{log_profile.model_path}\',"
        print_log(ensembles_str, level='INFO')

    path_of_ensembles = []
    for log_profile in model_profiles:
        path_of_ensembles.append(log_profile.model_path)

    return last_future_date.strftime('%Y-%m-%d'), path_of_ensembles


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
    
    # dir of logs
    parser.add_argument(
            '-ld', '--model_pool_log_dir',
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

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    configure_logger('search_by_logs', None, log_to_file=False)

    if args.top_percentage > 0:
        config["inference"]["top_percentage"] = args.top_percentage
    if args.max_ensemble_size > 0:
        config["inference"]["max_ensemble_size"] = args.max_ensemble_size
    if args.last_history_date != '':
        config["inference"]["last_history_date"] = args.last_history_date

    # If searching by logs, we can use the existing logs to find the best model
    print_log("Searching for the best model by existing logs...", level='INFO')
    # Get all model paths from the checkpoint directories via preconfigured requirements (min bonus, etc.)
    checkpoint_dirs = config['model_pool']['checkpoint_dirs']
    model_paths_available = get_all_model_paths_via_checkpoint_dirs(checkpoint_dirs, config)

    last_future_date, ensemble_paths = \
        search_ensemble_by_logs(
            model_pool_log_dir = args.model_pool_log_dir,
            config = config,
            last_history_date = config["inference"]["last_history_date"],
            top_percentage = config["inference"]["top_percentage"],
            num_to_select= config["inference"]["max_ensemble_size"],
            model_paths_available = model_paths_available)

    print_log(f"Last history date: {config['inference']['last_history_date']}, Last future date: {last_future_date}", level='INFO')
