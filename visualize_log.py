import argparse
import os
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

log_base = 1.01  # Base for logarithm, can be changed to any other value if needed

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyse log files.")
    parser.add_argument(
        "--log_path",
        type=str,
        default="",
        help="log file to analyse.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Log directory to analyse.",
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


# Custom formatter for y-axis ticks to show powers of 1.05
def power_formatter(value, pos):
    # Calculate exponent n where value ≈ log_base^n
    exponent = np.round(np.log(value) / np.log(log_base))
    return f'${log_base}^{{{int(exponent)}}}$'


def primary_process(single_log_path, log_dir):
    lines_of_all_files = []

    if single_log_path is not None:
        if not os.path.exists(single_log_path):
            #print(f"Log file {single_log_path} does not exist.")
            return None

        with open(single_log_path, "r") as f:
            lines = f.readlines()
            lines_of_all_files.append(lines)
    else:
        single_log_path = ''

    if log_dir is not None:
        if not os.path.exists(log_dir):
            #print(f"Log directory {log_dir} does not exist.")
            return None

        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        # sort log files by file name
        log_files.sort()

        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            if not os.path.exists(log_path):
                print(f"Log file {log_path} does not exist.")
                continue
            
            with open(log_path, "r") as f:
                lines = f.readlines()
                lines_of_all_files.append(lines)
    else:
        log_dir = ''

    date_list = None 
    total_amount_list_2d = []
    num_dates = 0

    for lines in lines_of_all_files:
        total_amount_list = []
        date_list_l = []
        for i in range(len(lines)):
            line = lines[i]
            if 'Date: ' in line:
                date_str = line.split()[-1].strip()
                next_line = lines[i + 1]

                # convert date string to date object if needed
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                date_list_l.append(date_obj)
                total_amount_list.append(float(next_line.split()[-1].strip()))
        
        if len(date_list_l) > num_dates:
            num_dates = len(date_list_l)

        if len(date_list_l) == 0 or len(date_list_l) < num_dates:
            continue

        date_list = date_list_l

        total_amount_list_2d.append(total_amount_list)

    return single_log_path + log_dir, date_list, total_amount_list_2d


def process_data(single_log_path, log_dir, logarithm, mean):
    result = primary_process(single_log_path, log_dir)

    if result is None:
        return None
    
    log_path, date_list, total_amount_list_2d = result

    # Convert total_amount_list to numpy array
    total_amount_list_2d = np.array(total_amount_list_2d)
    total_amount_list_2d /= total_amount_list_2d[:, 0:1]  # normalize by the first value

    # Get log scale of total amounts
    if logarithm:
        total_amount_list_2d = np.log(total_amount_list_2d + 1e-10) / np.log(log_base)   # Add a small constant to avoid log(0)

    if mean:
        # shape: (num_repeats, dates) to (1, dates)
        total_amount_list_2d = np.mean(total_amount_list_2d, axis=0, keepdims=True)

    total_amount_list_2d_max = np.max(total_amount_list_2d, keepdims=False)
    total_amount_list_2d_min = np.min(total_amount_list_2d, keepdims=False)

    value_top = total_amount_list_2d_max
    value_bottom = total_amount_list_2d_min

    # transpose the total_amount_list_2d to match date_list
    # (num_repeats, dates) or (1, dates) to (dates, num_repeats) or (dates, 1)
    total_amount_list_2d = total_amount_list_2d.T

    return log_path, value_top, value_bottom, date_list, total_amount_list_2d


def unify_data(data_group: list, logarithm):
    """
    Unify the data group to have the same number of dates.
    This function will truncate the longer sequences to match the length of the shortest one.
    """
    # Get common dates
    common_dates = set(data_group[0][3])
    for i in range(1, len(data_group)):
        common_dates.intersection_update(data_group[i][3])
    common_dates = sorted(common_dates)

    # Truncate each data entry to the common dates
    for i in range(len(data_group)):
        log_path, value_top, value_bottom, date_list, total_amount_list_2d = data_group[i]
        date_mask = np.isin(date_list, common_dates)
        # pickup elements from data_list where date_mask is True:
        total_amount_list_2d = total_amount_list_2d[date_mask]

        # We should rescale total_amount_list_2d
        if logarithm:
            # Just subtract the first value to zero if total_amount_list_2d is already logarithm format
            total_amount_list_2d -= total_amount_list_2d[0]
        else:
            # We should divide by the first value if total_amount_list_2d is not logarithm format
            total_amount_list_2d /= total_amount_list_2d[0]

        data_group[i] = (log_path, value_top, value_bottom, list(common_dates), total_amount_list_2d)

    return data_group


def plot_data(data_group : list, logarithm):

    plt.figure(figsize=(10, 5))
    title = f'Total Amounts Over Dates'
    all_value_top = -1e20
    all_value_bottom = 1e20

    legend_desc_list = []
    for log_path, value_top, value_bottom, date_list, total_amount_list_2d in data_group:
        plt.plot(date_list, total_amount_list_2d, marker='.', linestyle='-')

        legend_desc = os.path.basename(log_path.strip('/'))
        for i in range(total_amount_list_2d.shape[1]):
            if total_amount_list_2d.shape[1] > 1:
                legend_desc_list.append(f'Repeat {i + 1}')
            else:
                legend_desc_list.append(f'{legend_desc}')

        title += f'\n{log_path}'
        if value_top > all_value_top:
            all_value_top = value_top
        if value_bottom < all_value_bottom:
            all_value_bottom = value_bottom

    # Add legend for all plots
    plt.legend(legend_desc_list, loc='upper left')

    # Set x-axis to yyyy-mm-dd format
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))  # Set major ticks to every month
    plt.gca().xaxis.set_minor_locator(plt.matplotlib.dates.DayLocator(interval=1))  # Set minor ticks to every day
    #plt.tight_layout()
    plt.grid(True)
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability

    plt.xticks(rotation=90)
    plt.title(title)

    plt.xlabel('Date')

    if logarithm:
        plt.ylabel(f'Log {log_base} of Total Amount')

        # make all_value_bottom multiples of 5.0
        all_value_bottom = np.floor(all_value_bottom / 5.0) * 5.0
        # make all_value_top multiples of 5.0
        all_value_top = np.ceil(all_value_top / 5.0) * 5.0

        # print all_value_top and all_value_bottom:
        #print(f"Value Top: {all_value_top}")
        #print(f"Value Bottom: {all_value_bottom}")

        # Set y-axis value for each 1.0 increment
        plt.yticks(np.arange(all_value_bottom, all_value_top, 5.0),
                   [f'{i:.2f}' for i in np.arange(all_value_bottom, all_value_top, 5.0)])
        # Set minor y ticks
        plt.gca().yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator())
        
    else:
        plt.ylabel('Total Amount')

        # make all_value_bottom multiples of 0.05
        all_value_bottom = np.floor(all_value_bottom / 0.05) * 0.05
        # make all_value_top multiples of 0.05
        all_value_top = np.ceil(all_value_top / 0.05) * 0.05

        # print all_value_top and all_value_bottom:
        #print(f"Value Top: {all_value_top}")
        #print(f"Value Bottom: {all_value_bottom}")

        # Set y-axis value for each 0.05 increment
        plt.yticks(np.arange(all_value_bottom, all_value_top, 0.05),
                   [f'{i:.2f}' for i in np.arange(all_value_bottom, all_value_top, 0.05)])
        # Set minor y ticks
        plt.gca().yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator())
    
    # draw grid for minor ticks
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    plt.show()


def main():
    args = parse_args()
    single_log_path = args.log_path
    log_dir = args.log_dir
    logarithm = args.logarithm
    mean = args.mean

    single_log_path_list = single_log_path.split(',')
    log_dir_list = log_dir.split(',')

    data_group = []
    if len(single_log_path_list) > 0:
        for single_log_path in single_log_path_list:
            result = process_data(single_log_path, None, logarithm, mean)
            if result is not None:
                data_group.append(result)
    if len(log_dir_list) > 0:
        for log_dir in log_dir_list:
            result = process_data(None, log_dir, logarithm, mean)
            if result is not None:
                data_group.append(result)

    unify_data(data_group, logarithm)

    plot_data(data_group, logarithm)

if __name__ == "__main__":
    main()
