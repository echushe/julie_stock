from visualize_log import main
import argparse
import re
import random
import os

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyse log files.")
    
    parser.add_argument(
        "--model_pool_log_dir",
        type=str,
        default="",
        help="Log directory to analyse.",
    )

    parser.add_argument(
        "--num_dirs_to_select",
        type=int,
        default=20,
        help="Number of model directories to randomly select.",
    )

    # y axis log mode or not
    parser.add_argument(
        "--logarithm",
        action="store_true",
        help="Whether to use log scale for y-axis.",
    )

    parser.add_argument(
        "--mean",
        action="store_true",
        help="Enable mean output.",
    )
    return parser.parse_args()


def find_dir_of_each_model(model_pool_log_dir):
    import os
    model_log_dirs = []
    for item in os.listdir(model_pool_log_dir):
        # item should match this format: 
        if re.match(r'[0-9\-]+_[0-9\-]+_model[0-9]+', item):
            item_full_path = os.path.join(model_pool_log_dir, item)
            if os.path.isdir(item_full_path):
                model_log_dirs.append(item_full_path)
    return model_log_dirs


def randomly_select_dirs_from_model_log_dirs(model_log_dirs, num_dirs_to_select=5):
    if len(model_log_dirs) <= num_dirs_to_select:
        return model_log_dirs
    else:
        return random.sample(model_log_dirs, num_dirs_to_select)


if __name__ == "__main__":

    random.seed(int.from_bytes(os.urandom(4), 'big'))

    args = parse_args()

    model_pool_log_dir = args.model_pool_log_dir
    num_dirs_to_select = args.num_dirs_to_select
    logarithm = args.logarithm
    mean = args.mean
    
    model_log_dirs = find_dir_of_each_model(model_pool_log_dir)
    selected_dirs = randomly_select_dirs_from_model_log_dirs(model_log_dirs, num_dirs_to_select)
    selected_dirs = sorted(selected_dirs)

    selected_dirs_str = ','.join(selected_dirs)
    
    
    main('', selected_dirs_str, logarithm, mean)