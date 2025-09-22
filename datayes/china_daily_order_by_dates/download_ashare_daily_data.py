import requests
import time
import json
import os
import sys
import argparse
from datetime import datetime, timedelta


token = "000000000000000000000000000000000000000000000000000000000"

def read_ashare_tickers(file_path):

    lines = None
    with open(file_path, 'r') as f:
        lines = f.readlines()

    tickers = dict()
    for line in lines:
        items = line.split()
        tickers[items[0]] = items[1]

    #print(tickers)

    return tickers


def get_next_date(date):
    try:
        # Parse the input date string to a datetime object
        current_date = datetime.strptime(date, "%Y-%m-%d")
        
        # Add one day
        next_date = current_date + timedelta(days=1)
        
        # Format back to 'YYYY-MM-DD'
        return next_date.strftime("%Y-%m-%d")

    except ValueError as e:
        return f"Error: Invalid date format. Use 'YYYYMMDD'. ({e})"


def get_next_dates(date, num_days):

    try:
        current_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError as e:
        return f"Error: Invalid date format. Use 'YYYYMMDD'. ({e})"
    # Add next num_days
    # dates to the list
    next_dates = []
    for _ in range(num_days):
        current_date += timedelta(days=1)
        next_dates.append(current_date.strftime("%Y-%m-%d"))
    return next_dates


def download_all_tickers(asset_class='E'):
    url = '/api/master/getSecID.json?field=&assetClass={}&ticker=&partyID=&cnSpell=&listStatusCD=L,S,DE,UN&exchangeCD=XSHG,XSHE,XBEI'.format(asset_class)
    print(url)

    headers = {"Authorization": "Bearer " + token,
           "Accept-Encoding": "gzip, deflate"}
    
    request_success = False
    while not request_success:
        try:
            response = requests.request("GET",url='https://api.datayes.com/data/v1/' + url, headers=headers)
            request_success = True
        except TimeoutError as e:
            print(f"Error: {e}")
            time.sleep(5)  # Wait before retrying
        except ValueError as e:
            print(f"Value Error: {e}")
            time.sleep(5)  # Wait before retrying
        except Exception as e:
            print(f"Unexpected Error: {e}")
            time.sleep(5)  # Wait before retrying

    code = response.status_code
    response_as_str = response.content.decode('utf-8')
    response_as_dict = json.loads(response_as_str)

    if code != 200:
        raise ValueError('Error: {}'.format(response_as_str))
    
    if not "retCode" in response_as_dict or not "retMsg" in response_as_dict:
        raise ValueError('Error: {}'.format(response_as_str))
    
    if response_as_dict['retCode'] != 1 or response_as_dict['retMsg'] != 'Success':
        raise ValueError('Error: {}'.format(response_as_str))
    
    if not "data" in response_as_dict:
        raise ValueError('Error: {}'.format(response_as_str))
    
    tickers_id_as_key = dict()
    list_of_tickers = response_as_dict['data']
    for ticker_item in list_of_tickers:
        ticker_id = ticker_item['ticker']
        tickers_id_as_key[ticker_id] = ticker_item

    return tickers_id_as_key


def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data_as_dict = json.load(file)

    if not "data" in data_as_dict:
        print(f"Error: Invalid JSON file. 'data' key not found in {json_file_path}")
        sys.exit(1)
    
    return data_as_dict['data']


def download_multi_ticker_data_of_a_date(ticker_list, date, assert_class='E'):
    if assert_class not in ['E', 'IDX']:
        raise NotImplementedError('Error: asset_class {} is not supported'.format(assert_class))

    tickers_as_str = ','.join(ticker_list)

    if assert_class == 'E':
        url = '/api/market/getMktEqud.json?field=&beginDate=&endDate=&secID=&ticker={0}&tradeDate={1}'.format(tickers_as_str, date.replace("-", ""))
    elif assert_class == 'IDX':
        url = '/api/market/getMktIdxdExchange.json?field=&beginDate=&endDate=&ticker={0}&tradeDate={1}'.format(tickers_as_str, date.replace("-", ""))
    
    print(url)

    headers = {"Authorization": "Bearer " + token,
           "Accept-Encoding": "gzip, deflate"}
    
    response = requests.request("GET",url='https://api.datayes.com/data/v1/' + url, headers=headers)

    code = response.status_code
    response_as_str = response.content.decode('utf-8')
    response_as_dict = json.loads(response_as_str)

    if code != 200:
        raise ValueError('Error: {}'.format(response_as_str))
    
    if not "retCode" in response_as_dict or not "retMsg" in response_as_dict:
        raise ValueError('Error: {}, {}'.format(date, response_as_str))
    
    if response_as_dict['retCode'] != 1 or response_as_dict['retMsg'] != 'Success':
        raise ValueError('Error: {}, {}'.format(date, response_as_str))
    
    if not "data" in response_as_dict:
        raise ValueError('Error: {}, {}'.format(date, response_as_str))

    responses_as_list = response_as_dict['data']
    responses_ticker_as_key = dict()
    for response in responses_as_list:
        ticker = response['ticker']
        responses_ticker_as_key[ticker] = response
    
    return responses_ticker_as_key


def download_all_ticker_data_of_date(ticker_ids, date, asset_class='E'):
    if asset_class not in ['E', 'IDX']:
        raise NotImplementedError('Error: asset_class {} is not supported'.format(asset_class))

    ticker_ids_as_list = list(ticker_ids)
    ticker_ids_as_list.sort()

    print('Tickers to download: {}'.format(ticker_ids_as_list))

    responses_ticker_as_key = dict()

    idx = 0
    while idx < len(ticker_ids_as_list):
        time.sleep(1)
        print()

        if idx + 50 < len(ticker_ids_as_list):
            ticker_list = ticker_ids_as_list[idx:idx + 50]
        else:
            ticker_list = ticker_ids_as_list[idx:]

        try:
            new_responses_ticker_as_key = download_multi_ticker_data_of_a_date(ticker_list, date, asset_class)
        except TimeoutError as e:
            print(e)
            continue
        except ValueError as e:
            print(e)
            error_as_str = str(e)
            if 'Too many calls' in error_as_str:
                time.sleep(60)
            else:
                idx += 50
            continue
        except Exception as e:
            print(e)
            error_as_str = str(e)
            if 'Connection timed out' in error_as_str or \
                'Max retries exceeded with url' in error_as_str or \
                    'Failed to establish a new connection' in error_as_str:
                pass
            else:
                idx += 50           
            continue

        responses_ticker_as_key.update(new_responses_ticker_as_key)
        print('Downloaded tickers on {}'.format(date))
        idx += 50

    print()

    return responses_ticker_as_key



def download_one_day_data(tickers_id_as_key, date, root_dir, asset_class='E'):

    if asset_class not in ['E', 'IDX']:
        raise NotImplementedError('Error: asset_class {} is not supported'.format(asset_class))

    ticker_ids = set()
    for ticker_id, val in tickers_id_as_key.items():
        if 'listDate' in val and 'delistDate' in val:
            if date >= val['listDate'] and date <= val['delistDate']:
                ticker_ids.add(ticker_id)
        elif 'listDate' in val:
            if date >= val['listDate']:
                ticker_ids.add(ticker_id)
        elif 'delistDate' in val:
            if date <= val['delistDate']:
                ticker_ids.add(ticker_id)
        else:
            ticker_ids.add(ticker_id)

    if asset_class == 'E':
        root_dir = os.path.join(root_dir, 'ashare_daily_stock_order_by_dates')
    elif asset_class == 'IDX':
        root_dir = os.path.join(root_dir, 'ashare_daily_index_order_by_dates')
    else:
        raise NotImplementedError('Error: asset_class {} is not supported'.format(asset_class))
    
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    year = date[:4]
    year_dir = os.path.join(root_dir, year)
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)

    data_ticker_as_key = dict()
    if os.path.exists(os.path.join(year_dir, '{}.json'.format(date))):
        data_ticker_as_key.update(load_json_data(os.path.join(year_dir, '{}.json'.format(date))))
        ticker_ids_exist = set(data_ticker_as_key.keys())
        ticker_ids = ticker_ids - ticker_ids_exist
        print('Already downloaded {} tickers on {}'.format(len(ticker_ids_exist), date))
        print('Remaining {} tickers on {}'.format(len(ticker_ids), date))

    if len(ticker_ids) == 0:
        print('All tickers on {} are already downloaded'.format(date))
        return

    try:
        responses_ticker_as_key = download_all_ticker_data_of_date(ticker_ids, date, asset_class)
    except TimeoutError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)

    data_ticker_as_key.update(responses_ticker_as_key)
    data_to_save = dict()
    data_to_save['date'] = date
    data_to_save['data'] = data_ticker_as_key

    with open(os.path.join(year_dir, '{}.json'.format(date)), 'w') as f:
        f.write(json.dumps(data_to_save, indent=4))



def download_ashare_multi_day_data(first_date, last_date, root_dir, asset_class='E'):

    tickers_id_as_key = download_all_tickers(asset_class=asset_class)

    first_date = datetime.strptime(first_date, "%Y-%m-%d")
    last_date = datetime.strptime(last_date, "%Y-%m-%d")

    if first_date > last_date:
        print('Error: first date is later than last date')
        return

    current_date = first_date
    while current_date <= last_date:
        download_one_day_data(tickers_id_as_key, current_date.strftime("%Y-%m-%d"), root_dir=root_dir, asset_class=asset_class)
        current_date += timedelta(days=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='A sample script demonstrating argparse usage',
            epilog='Example: python script.py -n John --age 25'
        )
    
    parser.add_argument(
            '-f', '--first_date',
            type=str,
            help='first date of ashare ticker',
            default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        )
    parser.add_argument(
            '-l', '--last_date',
            type=str,
            help='last date of ashare ticker',
            default=datetime.now().strftime("%Y-%m-%d")
        )
    parser.add_argument(
            '-a', '--asset_class',
            type=str,
            help='asset class of ashare ticker',
            default='E'
        )
    parser.add_argument(
            '-r', '--root_dir',
            type=str,
            help='root directory to save ashare data',
            default='ashare_daily_data'
        )

    args = parser.parse_args()

    download_ashare_multi_day_data(args.first_date, args.last_date, args.root_dir, asset_class=args.asset_class)
