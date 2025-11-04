import time
import random
import argparse
from datayes.china_daily_order_by_dates.download_ashare_daily_data import download_ashare_multi_day_data
from datetime import datetime

parser = argparse.ArgumentParser(
        description='A sample script demonstrating argparse usage',
        epilog='Example: python script.py -n John --age 25'
    )

parser.add_argument(
        '-r', '--root_dir',
        type=str,
        help='root directory to save hk data',
        default='ashare_daily_data'
    )

args = parser.parse_args()

downloaded_mark_1 = False
downloaded_mark_2 = False
downloaded_mark_3 = False
date_of_last_download = '0000-00-00'

while True:

    # Get current date and time
    current_time = datetime.now()
    current_date_as_str = current_time.strftime("%Y-%m-%d")
    current_time_as_str = current_time.strftime("%H:%M:%S")
    print(f"AShare daily data endless run. Current date: {current_date_as_str}, Current time: {current_time_as_str}")

    if current_date_as_str != date_of_last_download:
        downloaded_mark_1 = False
        downloaded_mark_2 = False
        downloaded_mark_3 = False

    if (current_time.hour == 16 and not downloaded_mark_1) or \
       (current_time.hour == 20 and not downloaded_mark_2) or \
       (current_time.hour == 22 and not downloaded_mark_3):

        time.sleep(random.randint(0, 60))

        download_ashare_multi_day_data(
            first_date=current_date_as_str,
            last_date=current_date_as_str,
            root_dir=args.root_dir,
            asset_class='IDX')

        time.sleep(random.randint(0, 60))

        download_ashare_multi_day_data(
            first_date=current_date_as_str,
            last_date=current_date_as_str,
            root_dir=args.root_dir,
            asset_class='E')

        if current_time.hour == 16:
            downloaded_mark_1 = True
        elif current_time.hour == 20:
            downloaded_mark_2 = True
        elif current_time.hour == 22:
            downloaded_mark_3 = True

        date_of_last_download = current_date_as_str


    time.sleep(60)  # Sleep for 1 minute before checking again





    

