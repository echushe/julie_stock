import argparse
import os
import numpy as np

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Analyse log files.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="directory of the log files.",
    )
    parser.add_argument(
        "--last_trade_date",
        type=str,
        default="2025-05-28",
        help="last trade date to analyse.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    log_dir = args.log_dir

    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist.")
        return
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]

    all_lines = []
    for log_file in log_files:
        log_path = os.path.join(log_dir, log_file)
        if not os.path.exists(log_path):
            print(f"Log file {log_path} does not exist.")
            continue
        
        with open(log_path, "r") as f:
            lines = f.readlines()
            all_lines.extend(lines)

    final_total_amounts = []
    for i in range(len(all_lines)):
        line = all_lines[i]
        if 'Date: ' + args.last_trade_date in line:
            next_line = all_lines[i + 1]
            final_total_amounts.append(float(next_line.split()[-1]))

    print("Mean of final total amounts: ", sum(final_total_amounts) / len(final_total_amounts))
    themax = max(final_total_amounts)
    themin = min(final_total_amounts)
    #thestd = (sum((x - (sum(final_total_amounts) / len(final_total_amounts))) ** 2 for x in final_total_amounts) / len(final_total_amounts)) ** 0.5
    thestd = np.std(final_total_amounts)
    print("Max of final total amounts: ", themax)
    print("Min of final total amounts: ", themin)
    print("Std of final total amounts: ", thestd)

    themin =  9500000  #int(themin / 10000) * 10000
    themax = 19500000  #int(themax / 10000 + 1) * 10000

    n_ranges = 100

    each_range = (themax - themin) / n_ranges
    thresholds = [themin + i * each_range for i in range(n_ranges + 1)]

    count_ranges = [0] * n_ranges
    for amount in final_total_amounts:
        for i in range(n_ranges):
            if thresholds[i] <= amount < thresholds[i + 1]:
                count_ranges[i] += 1
                break

    # Print the count of final total amounts in each range
    print("Count of final total amounts in each range:")
    #for i in range(n_ranges):
    #    print(f"{thresholds[i] / 10000:.2f} - {thresholds[i + 1] / 10000:.2f}: {count_ranges[i]}")
    # Graphically display the count of final total amounts in each range
    import matplotlib.pyplot as plt
    plt.bar([f"{thresholds[i] / 10000:.2f} - {thresholds[i + 1] / 10000:.2f}" for i in range(n_ranges)], count_ranges)
    plt.xlabel('Final Total Amount Ranges')
    plt.ylabel('Count')
    plt.title('Count of Final Total Amounts in Each Range')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.ylim(0, 70)
    plt.show()


main()
