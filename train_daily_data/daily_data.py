import os
import json
import numpy as np
import cv2
import datetime

from pytz import timezone
from typing import Union
from train_daily_data.utils import milliseconds_to_datetime, milliseconds_to_date, date_to_milliseconds, next_date_of_date
from train_daily_data.global_logger import print_log

from train_daily_data.BEI_code_mapping import BEICodeMapping

ASHARE_STOCK = 0
NASDAQ_STOCK = 1
ASHARE_INDEX = 2
NASDAQ_INDEX = 3
class KlineAgumentNames:

    def __init__(self, type=ASHARE_STOCK):

        self.type = type

        self.open_price = {
            ASHARE_STOCK: 'openPrice',
            NASDAQ_STOCK: 'o',
            ASHARE_INDEX: 'openIndex',
        }

        self.close_price = {
            ASHARE_STOCK: 'closePrice',
            NASDAQ_STOCK: 'c',
            ASHARE_INDEX: 'closeIndex',
        }

        self.highest_price = {
            ASHARE_STOCK: 'highestPrice',
            NASDAQ_STOCK: 'h',
            ASHARE_INDEX: 'highestIndex',
        }

        self.lowest_price = {
            ASHARE_STOCK: 'lowestPrice',
            NASDAQ_STOCK: 'l',
            ASHARE_INDEX: 'lowestIndex',
        }

        self.turnover_vol = {
            ASHARE_STOCK: 'turnoverVol',
            NASDAQ_STOCK: 'v',
            ASHARE_INDEX: 'turnoverVol',
        }

        self.turnover_value = {
            ASHARE_STOCK: 'turnoverValue',
            #NASDAQ_STOCK: 'v',
            ASHARE_INDEX: 'turnoverValue',
        }

    def get_open_price_name(self):
        if self.type in self.open_price:
            return self.open_price[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.open_price.keys()}.")
        
    def get_close_price_name(self):
        if self.type in self.close_price:
            return self.close_price[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.close_price.keys()}.")

    def get_highest_price_name(self):
        if self.type in self.highest_price:
            return self.highest_price[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.highest_price.keys()}.")

    def get_lowest_price_name(self):
        if self.type in self.lowest_price:
            return self.lowest_price[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.lowest_price.keys()}.")

    def get_turnover_vol_name(self):
        if self.type in self.turnover_vol:
            return self.turnover_vol[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.turnover_vol.keys()}.")

    def get_turnover_value_name(self):
        if self.type in self.turnover_value:
            return self.turnover_value[self.type]
        else:
            raise ValueError(f"Invalid type: {self.type}. Supported types are {self.turnover_value.keys()}.") 


class StockDataPreprocessor:
    def __init__(
            self,
            klines_as_dict_date_as_key,
            market : str,
            start_date : str,
            end_date : str,
            index_data : bool=False,
            min_daily_turnover_value=0.0,
            strong_decrease_ignore_threshold=-0.2,
            strong_increase_ignore_threshold=0.25,
            strong_fluctuation_ignore_threshold=0.3,
            data_visual_dir='data_visualization',
            ) -> None:
        
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        if index_data:
            self.data_type = 'index'
        else:
            self.data_type = 'stock'       
        self.data_visual_dir = data_visual_dir

        self.all_tickers = set()
        self.all_trade_dates = set()
        #self.samples = []

        klines_as_dict_ticker_as_key = self.__date_as_key_to_ticker_as_key(klines_as_dict_date_as_key)

        self.__mark_unusual_dates_for_tickers(
            klines_as_dict_ticker_as_key,
            min_daily_turnover_value,
            strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold)
        
        self.__visualize_trading_dates_for_tickers(klines_as_dict_date_as_key)
        self.__visualize_trading_dates_for_preprocessed_tickers(klines_as_dict_ticker_as_key)

        self.klines_as_dict_date_as_key = klines_as_dict_date_as_key
        self.klines_as_dict_ticker_as_key = klines_as_dict_ticker_as_key
        

    def __visualize_trading_dates_for_tickers(self, klines_as_dict_date_as_key):

        # Print stack information of this method
        print_log(
            f"Visualizing trading dates for tickers...",
            level='INFO')       

        img_width = 0
        img_height = len(self.all_tickers)

        all_trade_dates_sorted = sorted(list(self.all_trade_dates))
        all_tickers_sorted = sorted(list(self.all_tickers))

        first_date = all_trade_dates_sorted[0]
        last_date = all_trade_dates_sorted[-1]

        next_date = first_date
        while next_date <= last_date:
            img_width += 1
            next_date = next_date_of_date(next_date)

        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        next_date = first_date
        date_idx = 0
        while next_date <= last_date:
            print_log(next_date, level='DEBUG')
            if next_date in klines_as_dict_date_as_key:
                available_tickers = klines_as_dict_date_as_key[next_date]
            else:
                available_tickers = dict()

            for ticker_idx, ticker in enumerate(all_tickers_sorted):
                if ticker in available_tickers:
                    img[ticker_idx, date_idx] = 255

            next_date = next_date_of_date(next_date)
            date_idx += 1
        print_log('', level='DEBUG')

        if not os.path.exists(self.data_visual_dir):
            os.makedirs(self.data_visual_dir)
        img_path = os.path.join(self.data_visual_dir, '{}_{}_{}_{}.png'.format(self.market, self.data_type, self.start_date, self.end_date))
        cv2.imwrite(img_path, img)

        print_log(
            f"Visualizing trading dates for tickers... Done. Image saved to {img_path}",
            level='INFO')


    def __visualize_trading_dates_for_preprocessed_tickers(self, klines_as_dict_ticker_as_key):

        # Print stack information of this method
        print_log(f"Visualizing trading dates for preprocessed tickers...", level='INFO')

        img_width = 0
        img_height = len(self.all_tickers)

        all_trade_dates_sorted = sorted(list(self.all_trade_dates))
        all_tickers_sorted = sorted(list(self.all_tickers))

        first_date = all_trade_dates_sorted[0]
        last_date = all_trade_dates_sorted[-1]

        next_date = first_date
        while next_date <= last_date:
            img_width += 1
            next_date = next_date_of_date(next_date)

        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for ticker_idx, ticker in enumerate(all_tickers_sorted):
            print_log(ticker, level='DEBUG')
            if ticker in klines_as_dict_ticker_as_key:
                kline_date_as_key = klines_as_dict_ticker_as_key[ticker]
            else:
                kline_date_as_key = dict()

            date_idx = 0
            next_date = first_date
            while next_date <= last_date:
                if next_date in kline_date_as_key:
                    kline_item = kline_date_as_key[next_date]
                    if kline_item['unusual'] == True:
                        img[ticker_idx, date_idx, 2] = 255  
                    else:
                        img[ticker_idx, date_idx] = 255
                else:
                    img[ticker_idx, date_idx] = (0, 0, 0)

                next_date = next_date_of_date(next_date)
                date_idx += 1
        print_log('', level='DEBUG')

        if not os.path.exists(self.data_visual_dir):
            os.makedirs(self.data_visual_dir)
        img_path = os.path.join(self.data_visual_dir, '{}_{}_{}_{}_preprocessed.png'.format(self.market, self.data_type, self.start_date, self.end_date))
        cv2.imwrite(img_path, img)

        print_log(
            f"Visualizing trading dates for preprocessed tickers... Done. Image saved to {img_path}",
            level='INFO')

    def __date_as_key_to_ticker_as_key(self, klines_as_dict_date_as_key):

        klines_as_dict_ticker_as_key = dict()

        for trade_date, tickers in klines_as_dict_date_as_key.items():
            for ticker in tickers.keys():
                self.all_trade_dates.add(trade_date)
                self.all_tickers.add(ticker)

        # We assume that all these recorded trade dates covers all official trade dates of the stock market
        # We need to sort the trade dates
        all_trade_dates_sorted = sorted(list(self.all_trade_dates))

        # We need to sort the tickers
        all_tickers_sorted = sorted(list(self.all_tickers))

        #missing_trading_dates_ticker_as_key = dict()
        missing_trading_dates_date_as_key = dict()

        for ticker in all_tickers_sorted:
            kline = self.__find_kline_for_ticker(ticker, klines_as_dict_date_as_key, all_trade_dates_sorted)
            klines_as_dict_ticker_as_key[ticker] = kline

        return klines_as_dict_ticker_as_key


    def __find_kline_for_ticker(self, ticker, klines_as_dict_date_as_key, all_trade_dates_sorted):

        print_log(f"Interpolating missing data for {ticker}...", level='DEBUG')

        # Create a new dictionary to store the kline for this ticker
        kline = dict()

        # Iterate through all trade dates
        for trade_date in all_trade_dates_sorted:
            if ticker in klines_as_dict_date_as_key[trade_date]:
                # If the ticker exists for this trade date, use its data
                kline[trade_date] = klines_as_dict_date_as_key[trade_date][ticker]

            #else:
            #    missing_trading_dates.append(trade_date)

        return kline


    def __mark_unusual_dates_for_tickers(
            self,
            klines_as_dict_ticker_as_key,
            min_daily_turnover_value,
            strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold):
        # We need to sort the tickers
        all_tickers_sorted = sorted(list(self.all_tickers))
        # Iterate through all trade dates
        for ticker in all_tickers_sorted:
            for trade_date, kline_item in klines_as_dict_ticker_as_key[ticker].items():
                # Calculate the percentage change
                open_price = kline_item['openPrice']
                close_price = kline_item['closePrice']
                highest_price = kline_item['highestPrice']
                lowest_price = kline_item['lowestPrice']
                turnover_vol = kline_item['turnoverVol']
                turnover_value = kline_item['turnoverValue']

                if open_price <= 0 or close_price <= 0 or highest_price <= 0 or lowest_price <= 0 or turnover_vol <= 0:
                    # If any of the prices are less than or equal to 0, we can ignore this date
                    print_log(f"Invalid price data for {ticker} on {trade_date}: openPrice={open_price}, closePrice={close_price}, highestPrice={highest_price}, lowestPrice={lowest_price}, turnoverVol={turnover_vol}",
                              level='DEBUG')
                    kline_item['unusual'] = True
                    continue

                if 'preClosePrice' in kline_item:
                    pre_close_price = kline_item['preClosePrice']
                    if pre_close_price <= 0:
                        # If the pre-close price is less than or equal to 0, we can ignore this date
                        print_log(f"Invalid pre-close price for {ticker} on {trade_date}: preClosePrice={pre_close_price}",
                                  level='DEBUG')
                        kline_item['unusual'] = True
                        continue
                if 'actPreClosePrice' in kline_item:
                    act_pre_close_price = kline_item['actPreClosePrice']
                    if act_pre_close_price <= 0:
                        # If the actual pre-close price is less than or equal to 0, we can ignore this date
                        print_log(f"Invalid actual pre-close price for {ticker} on {trade_date}: actPreClosePrice={act_pre_close_price}",
                                  level='DEBUG')
                        kline_item['unusual'] = True
                        continue

                percentage_change = (close_price - open_price) / open_price
                percentage_fluctuation = (highest_price - lowest_price) / open_price

                if turnover_value < min_daily_turnover_value:
                    # If the turnover value is less than the minimum daily turnover value, we can ignore this date
                    print_log(f"Low turnover value: {ticker} on {trade_date}: {turnover_value}", level='DEBUG')
                    kline_item['unusual'] = True

                elif percentage_change < strong_decrease_ignore_threshold:
                    # Mark this date as unusual
                    print_log(f"Strong decrease: {ticker} on {trade_date}: {percentage_change}", level='DEBUG')
                    kline_item['unusual'] = True
                elif percentage_change > strong_increase_ignore_threshold:
                    # Mark this date as unusual
                    print_log(f"Strong increase: {ticker} on {trade_date}: {percentage_change}", level='DEBUG')
                    kline_item['unusual'] = True

                elif percentage_fluctuation > strong_fluctuation_ignore_threshold:
                    # Mark this date as unusual
                    print_log(f"Strong fluctuation: {ticker} on {trade_date}: {percentage_fluctuation}", level='DEBUG')
                    kline_item['unusual'] = True


class StockData:

    def __init__(self, 
                root_dir, 
                market : str, 
                start_date : str, 
                end_date : str, 
                index_data : bool=False,
                exchange_cd = None,
                tickers : Union[set, list, tuple]=[],
                tickers_to_exclude : Union[set, list, tuple]=[],
                min_daily_turnover_value=0.0,
                strong_decrease_ignore_threshold=-0.2,
                strong_increase_ignore_threshold=0.25,
                strong_fluctuation_ignore_threshold=0.3,
                ) -> None:


        self.nasdaq_timezone = timezone('US/Eastern')
        self.ashare_timezone = timezone('Asia/Shanghai')

        print_log(f"Loading dataset for {tickers} from {start_date} to {end_date}", level='INFO')

        if market == 'nasdaq':
            if index_data:
                self.kline_arg_names = KlineAgumentNames(type=NASDAQ_INDEX)
            else:
                self.kline_arg_names = KlineAgumentNames(type=NASDAQ_STOCK)
        elif market in {'ashare', 'hk'}:
            if index_data:
                self.kline_arg_names = KlineAgumentNames(type=ASHARE_INDEX)
            else:
                self.kline_arg_names = KlineAgumentNames(type=ASHARE_STOCK)

        if exchange_cd is not None and exchange_cd == '':
            exchange_cd = None
        self.exchange_cd = exchange_cd

        if tickers is None:
            tickers = []
        if tickers_to_exclude is None:
            tickers_to_exclude = []
        
        klines_as_dict_date_as_key = dict()
        names_ticker_as_key = dict()

        if market == 'nasdaq':
            if len(tickers) == 0:
                # If tickers is None or empty, load all json files in the root_dir
                json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
                for json_file in json_files:
                    to_ignore = False
                    for t_h in tickers_to_exclude:
                        if json_file.startswith(t_h):
                            print_log(f"Warning: File {json_file} is excluded from loading.", level='DEBUG')
                            to_ignore = True
                            break
                    if to_ignore:
                        continue
                    self.__load_klines_from_json(klines_as_dict_date_as_key, market, root_dir, json_file, start_date, end_date)
            else:
                if not isinstance(tickers, (set, list, tuple)):
                    raise ValueError(f"tickers should be a set, list or tuple. Got {type(tickers)}")
                for ticker in tickers:
                    to_ignore = False
                    for t_h in tickers_to_exclude:
                        if ticker.startswith(t_h):
                            print_log(f"Warning: Ticker {ticker} is excluded from loading.", level='DEBUG')
                            to_ignore = True
                            break
                    if to_ignore:
                        continue
                    json_file = ticker + '.json'
                    if not os.path.exists(os.path.join(root_dir, json_file)):
                        print_log(f"Warning: File {json_file} does not exist in {root_dir}", level='WARNING')
                        continue
                    self.__load_klines_from_json(klines_as_dict_date_as_key, market, root_dir, json_file, start_date, end_date)
        elif market in {'ashare', 'hk'}:
            self.__load_klines_from_ashare_stock_order_by_dates(klines_as_dict_date_as_key, names_ticker_as_key, root_dir, start_date, end_date, tickers, tickers_to_exclude)
            klines_as_dict_date_as_key = self.__map_old_BEI_code_to_new_code(klines_as_dict_date_as_key)

        # Interpolate missing data
        self.stock_data_preprocessor = StockDataPreprocessor(
            klines_as_dict_date_as_key,
            market=market,
            start_date=start_date,
            end_date=end_date,
            index_data=index_data,
            min_daily_turnover_value=min_daily_turnover_value,
            strong_decrease_ignore_threshold=strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold=strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold=strong_fluctuation_ignore_threshold,)
    
        self.klines_as_dict_date_as_key = self.stock_data_preprocessor.klines_as_dict_date_as_key
        self.klines_as_dict_ticker_as_key = self.stock_data_preprocessor.klines_as_dict_ticker_as_key
        self.names_ticker_as_key = names_ticker_as_key

        
    def __load_klines_from_ashare_stock_order_by_dates(self, klines_as_dict_date_as_key, names_ticker_as_key, root_dir, start_date, end_date, tickers=[], tickers_to_exclude=[]):
        # Go through each day in the date range
        # and load the corresponding data
        warning_msg = ''

        # Get date of today as YYYY-MM-DD:
        today = datetime.date.today().strftime('%Y-%m-%d')

        date = start_date
        while date <= end_date and date <= today:
            year_str = date.split('-')[0]
            json_dir = os.path.join(root_dir, year_str)
            if not os.path.exists(json_dir):
                print_log(f"Warning: Directory {json_dir} does not exist.", level='DEBUG')
            else:
                json_file_path = os.path.join(json_dir, date + '.json')
                if not os.path.exists(json_file_path):
                    print_log(f"Warning: File {json_file_path} does not exist.", level='WARNING')
                else:
                    self.__load_tickers_from_json_of_date(klines_as_dict_date_as_key, names_ticker_as_key, json_file_path, date, tickers, tickers_to_exclude)

            date = next_date_of_date(date)


    def __load_tickers_from_json_of_date(self, klines_as_dict_date_as_key, names_ticker_as_key, json_file_path, date, tickers=[], tickers_to_exclude=[]):
        print_log(f"Loading trading data from {json_file_path}", level='DEBUG')
        with open(json_file_path) as f:
            data_as_str = f.read()

            data_as_dict = json.loads(data_as_str)

            if 'data' not in data_as_dict:
                print_log(f"Error: Invalid JSON file. 'data' key not found in {json_file_path}", level='ERROR')
                return

            tickers_as_dict = data_as_dict['data']

            if len(tickers) == 0:
                tickers = list(tickers_as_dict.keys())

            for ticker in tickers:

                to_ignore = False
                for t_h in tickers_to_exclude:
                    if ticker.startswith(t_h):
                        print_log(f"Warning: Ticker {ticker} is excluded from loading.", level='DEBUG')
                        to_ignore = True
                        break

                if to_ignore:
                    continue

                if ticker not in tickers_as_dict:
                    print_log(f"Ticker {ticker} not found in {json_file_path}", level='DEBUG')
                    continue

                kline_item = tickers_as_dict[ticker]

                if self.exchange_cd is not None and kline_item['exchangeCD'] != self.exchange_cd:
                    continue

                if 'secShortName' in kline_item and ticker not in names_ticker_as_key:
                    # If the ticker is not in names_ticker_as_key, we can add it
                    ticker_ch_name = kline_item['secShortName']
                    names_ticker_as_key[ticker] = ticker_ch_name
                    print_log(f"Ticker {ticker} name is {kline_item['secShortName']}", level='DEBUG')

                try:
                    unified_kline_item = {
                        'openPrice': kline_item[self.kline_arg_names.get_open_price_name()],
                        'closePrice': kline_item[self.kline_arg_names.get_close_price_name()],
                        'highestPrice': kline_item[self.kline_arg_names.get_highest_price_name()],
                        'lowestPrice': kline_item[self.kline_arg_names.get_lowest_price_name()],
                        'tickerName': kline_item['secShortName'],
                        'unusual': False
                    }

                    if self.kline_arg_names.get_turnover_vol_name() in kline_item:
                        # If the turnover volume is available, use it
                        unified_kline_item['turnoverVol'] = kline_item[self.kline_arg_names.get_turnover_vol_name()]
                    else:

                        if self.kline_arg_names.get_turnover_value_name() in kline_item:
                            # If the turnover value is available, we can calculate the turnover volume
                            unified_kline_item['turnoverVol'] = \
                                kline_item[self.kline_arg_names.get_turnover_value_name()] / (unified_kline_item['closePrice'] + 1e-6) # Avoid division by zero
                        else:
                            unified_kline_item['turnoverVol'] = 1000 # Default value if not available

                    if self.kline_arg_names.get_turnover_value_name() in kline_item:
                        # If the turnover value is available, use it
                        unified_kline_item['turnoverValue'] = kline_item[self.kline_arg_names.get_turnover_value_name()]
                    else:
                        unified_kline_item['turnoverValue'] = unified_kline_item['turnoverVol'] * unified_kline_item['closePrice'] # Default value if not available

                    if 'preClosePrice' in kline_item:
                        unified_kline_item['preClosePrice'] = kline_item['preClosePrice']
                    if 'actPreClosePrice' in kline_item:
                        unified_kline_item['actPreClosePrice'] = kline_item['actPreClosePrice']

                except KeyError as e:
                    print_log(f"Error: Key {e} not found in {json_file_path} for ticker {ticker}", level='ERROR')
                    continue

                if date not in klines_as_dict_date_as_key:
                    klines_as_dict_date_as_key[date] = {ticker : unified_kline_item}
                else:
                    klines_as_dict_date_as_key[date][ticker] = unified_kline_item


    def __load_klines_from_json(self, klines_as_dict_date_as_key, market, root_dir, json_file, start_date, end_date):
        print_log(f"Loading trading data from {json_file}", level='INFO')
        with open(os.path.join(root_dir, json_file)) as f:
            data_as_str = f.read()

            data_as_dict = json.loads(data_as_str)

            if market == 'nasdaq':

                if 'results' not in data_as_dict:
                    print_log(f"Error: Invalid JSON file. 'results' key not found in {json_file}", level='ERROR')
                    return

                kline_as_dict = data_as_dict['results']
                kline_as_list = sorted(kline_as_dict.values(), key=lambda x: x['t'])

                start_t = date_to_milliseconds(start_date, self.nasdaq_timezone)
                end_t = date_to_milliseconds(end_date, self.nasdaq_timezone)

                ticker = json_file.strip('.json')
                self.__add_nasdaq_klines_to_dict(klines_as_dict_date_as_key, kline_as_list, ticker, start_t, end_t)

            elif market in {'ashare', 'hk'}:
                if 'data' not in data_as_dict:
                    print_log(f"Error: Invalid JSON file. 'data' key not found in {json_file}", level='ERROR')
                    return

                kline_as_dict = data_as_dict['data']
                kline_as_list = sorted(kline_as_dict.values(), key=lambda x: x['tradeDate'])

                ticker = json_file.split('_')[0]
                self.__add_ashare_klines_to_dict(klines_as_dict_date_as_key, kline_as_list, ticker, start_date, end_date)
            
            else:
                print_log(f"Error: Invalid market type. Supported markets are 'nasdaq' and 'ashare'.", level='ERROR')
                raise ValueError(f"Invalid market type: {market}")
    

    def __add_nasdaq_klines_to_dict(self, klines_as_dict_date_as_key, kline_as_list, ticker, start_t, end_t):
        '''
        kline_as_list = sorted(kline_as_list, key=lambda x: x['t'])
        start_t = date_to_milliseconds(start_date, self.nasdaq_timezone)
        end_t = date_to_milliseconds(end_date, self.nasdaq_timezone)
        '''
        for kline_item in kline_as_list:

            if kline_item['t'] < start_t or kline_item['t'] > end_t:
                continue
            
            try:
                unified_kline_item = {
                    'openPrice': kline_item[self.kline_arg_names.get_open_price_name()],
                    'closePrice': kline_item[self.kline_arg_names.get_close_price_name()],
                    'highestPrice': kline_item[self.kline_arg_names.get_highest_price_name()],
                    'lowestPrice': kline_item[self.kline_arg_names.get_lowest_price_name()],
                    'turnoverVol': kline_item[self.kline_arg_names.get_turnover_vol_name()],
                    'unusual': False
                }
            except KeyError as e:
                print_log(f"Error: Key {e} not found in for ticker {ticker}", level='ERROR')
                continue

            trade_date = milliseconds_to_date(kline_item['t'], self.nasdaq_timezone)
            if trade_date not in klines_as_dict_date_as_key:
                klines_as_dict_date_as_key[trade_date] = {ticker : unified_kline_item}
            else:
                klines_as_dict_date_as_key[trade_date][ticker] = unified_kline_item



    def __add_ashare_klines_to_dict(self, klines_as_dict_date_as_key, kline_as_list, ticker, start_date, end_date):
        '''
        kline_as_list = sorted(kline_as_list, key=lambda x: x['tradeDate'])
        start_t = date_to_milliseconds(start_date, self.ashare_timezone)
        end_t = date_to_milliseconds(end_date, self.ashare_timezone)
        '''
        for kline_item in kline_as_list:

            if kline_item['tradeDate'] < start_date or kline_item['tradeDate'] > end_date:
                continue

            if self.exchange_cd is not None and kline_item['exchangeCD'] != self.exchange_cd:
                continue

            try:
                unified_kline_item = {
                    'openPrice': kline_item[self.kline_arg_names.get_open_price_name()],
                    'closePrice': kline_item[self.kline_arg_names.get_close_price_name()],
                    'highestPrice': kline_item[self.kline_arg_names.get_highest_price_name()],
                    'lowestPrice': kline_item[self.kline_arg_names.get_lowest_price_name()],
                    'turnoverVol': kline_item[self.kline_arg_names.get_turnover_vol_name()],
                    'unusual': False
                }
            except KeyError as e:
                print_log(f"Error: Key {e} not found in for ticker {ticker}", level='ERROR')
                continue

            trade_date = kline_item['tradeDate']
            if trade_date not in self.klines_as_dict_date_as_key:
                klines_as_dict_date_as_key[trade_date] = {ticker : unified_kline_item}
            else:
                klines_as_dict_date_as_key[trade_date][ticker] = unified_kline_item


    def __map_old_BEI_code_to_new_code(self, klines_as_dict_date_as_key):
        # Create a BEICodeMapping instance
        bei_code_mapping = BEICodeMapping()

        # Create a new dictionary to store the updated klines
        updated_klines_as_dict_date_as_key = dict()

        for trade_date, tickers in klines_as_dict_date_as_key.items():
            updated_tickers = dict()
            for ticker, kline_item in tickers.items():
                new_ticker = bei_code_mapping.get_new_code(ticker)
                if new_ticker != ticker:
                    print_log(f"Mapping old BEI code {ticker} to new code {new_ticker}", level='DEBUG')
                updated_tickers[new_ticker] = kline_item
            updated_klines_as_dict_date_as_key[trade_date] = updated_tickers

        # Replace the old klines with the updated klines
        return updated_klines_as_dict_date_as_key


    def __go_through_klines_for_debug(self, index_ticker):
            # Go through the index klines and print the trade dates
            all_trade_dates = sorted(list(self.klines_as_dict_date_as_key.keys()))

            for i in range(len(all_trade_dates)):
                if i < 0 or i >= len(all_trade_dates):
                    continue
                current_trade_date = all_trade_dates[i]
                klines_ticker_as_key = self.klines_as_dict_date_as_key[current_trade_date]
                if index_ticker not in klines_ticker_as_key:
                    print_log(f"Index ticker {index_ticker} is not in the klines of trade date {current_trade_date}.", level='INFO')
                    continue
                kline_item = klines_ticker_as_key[index_ticker]
                if kline_item['unusual'] == True:
                    print_log(f"Trade date {current_trade_date} is marked as unusual for index ticker {index_ticker}.", level='INFO')
                    continue
    

if __name__ == '__main__':
    # Example usage
    root_dir = 'datayes_data_sample/hk_daily_index_order_by_dates'
    exchange_cd= 'XHKG'
    #root_dir = '../../data/us_daily_data'
    market = 'hk'
    #market = 'nasdaq'
    start_date = '2000-01-01'
    end_date = '2024-12-31'
    ticker = None

    data = StockData(   
        root_dir, market, start_date, end_date,
        index_data=True,
        tickers=None,
        exchange_cd=exchange_cd,
        strong_decrease_ignore_threshold=-0.8,
        strong_increase_ignore_threshold=5,
        strong_fluctuation_ignore_threshold=10)

    print("Press Enter to exit...")
    input()




