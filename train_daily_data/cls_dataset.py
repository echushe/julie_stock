import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import Dataset
from typing import Union

from train_daily_data.daily_data import StockData
from train_daily_data.cls_rules import EarningRateCLSRule
from train_daily_data.global_logger import print_log, configure_logger

class ClsDataset(Dataset):

    def __init__(self, 
                stock_root_dir,
                index_root_dir,
                market : str, 
                start_date : str, 
                end_date : str,
                exchange_cd : str=None, 
                stock_tickers : Union[set, list, tuple]=[],
                stock_tickers_to_exclude : Union[set, list, tuple]=[],
                index_tickers : Union[set, list, tuple]=[],
                index_tickers_to_exclude : Union[set, list, tuple]=[],
                min_daily_turnover_value=0.0,
                strong_decrease_ignore_threshold=-0.2,
                strong_increase_ignore_threshold=0.25,
                strong_fluctuation_ignore_threshold=0.3,
                input_t_len=10,
                target_t_len=2,
                cls_rule=None,
                n_classes=5
                ) -> None:

        super(ClsDataset, self).__init__()


        self.stock_data = StockData(
            root_dir=stock_root_dir,
            market=market,
            start_date=start_date,
            end_date=end_date,
            index_data=False,
            exchange_cd=exchange_cd,
            tickers=stock_tickers,
            tickers_to_exclude=stock_tickers_to_exclude,
            min_daily_turnover_value=min_daily_turnover_value,
            strong_decrease_ignore_threshold=strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold=strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold=strong_fluctuation_ignore_threshold)
        

        self.index_data = StockData(
            root_dir=index_root_dir,
            market=market,
            start_date=start_date,
            end_date=end_date,
            index_data=True,
            exchange_cd=exchange_cd,
            tickers=index_tickers,
            tickers_to_exclude=index_tickers_to_exclude,
            strong_decrease_ignore_threshold=strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold=strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold=strong_fluctuation_ignore_threshold)
        

        self.__fix_trade_dates(self.index_data, self.stock_data)

        #self.multiprocessing_manager = multiprocessing.Manager()

        self.index_samples_date_as_key = self.__generate_index_samples(
            self.index_data,
            input_t_len=input_t_len,
            target_t_len=target_t_len)

        self.stock_samples, self.stock_samples_date_as_key = self.__generate_stock_samples(
            self.stock_data,
            input_t_len=input_t_len,
            target_t_len=target_t_len)


        if cls_rule is None:
            if n_classes is None or n_classes <= 1:
                raise ValueError("n_classes must be a integer larger than 1.")
            print_log(f"Auto-determining the classification rule with {n_classes} classes...", level='INFO')
            self.cls_rule = EarningRateCLSRule()
            cls_threshold_list = self.__auto_determine_cls_rule(n_classes=n_classes)
            self.cls_rule.cls_threshold_list = cls_threshold_list
        else:
            self.cls_rule = EarningRateCLSRule(cls_threshold_list=cls_rule)

        # Transform list and dict into managed list and dict
        #self.stock_samples = self.multiprocessing_manager.list(self.stock_samples)
        #self.stock_samples_date_as_key = self.multiprocessing_manager.dict(self.stock_samples_date_as_key)


    def __fix_trade_dates(self, index_data, stock_data):
        """
        Fix the trade dates of the index data and stock data.
        The trade dates of the index data and stock data should be the same.
        """
        index_trade_dates = set(index_data.klines_as_dict_date_as_key.keys())
        stock_trade_dates = set(stock_data.klines_as_dict_date_as_key.keys())

        all_trade_dates = index_trade_dates.union(stock_trade_dates)

        for trade_date in all_trade_dates:
            if trade_date not in index_data.klines_as_dict_date_as_key:
                # If the trade date is not in the index data, add it with empty data
                print_log(f"Trade date {trade_date} is not in index data, adding empty data.", level='DEBUG')
                index_data.klines_as_dict_date_as_key[trade_date] = dict()
            if trade_date not in stock_data.klines_as_dict_date_as_key:
                # If the trade date is not in the stock data, add it with empty data
                print_log(f"Trade date {trade_date} is not in stock data, adding empty data.", level='DEBUG')
                stock_data.klines_as_dict_date_as_key[trade_date] = dict()
    

    def __generate_index_samples_no_targets(self, index_data, input_t_len=10):
        # Convert the klines_as_dict_ticker_as_key into training samples

        # Sort the trade dates
        trade_dates_sorted = sorted(list(index_data.klines_as_dict_date_as_key.keys()))

        index_samples_date_as_key = dict()
        for index_ticker, kline_date_as_key in index_data.klines_as_dict_ticker_as_key.items():
            print_log(f"Converting index {index_ticker} into training samples...", level='INFO')

            # Iterate through all trade dates
            for idx in range(len(trade_dates_sorted) - input_t_len + 1):

                skip_this_sample_window = False
                # Check if there are any unusual dates in the sample window
                for k in range(input_t_len):
                    trade_date = trade_dates_sorted[idx + k]
                    if trade_date not in kline_date_as_key:
                        # If the trade date is not in the kline_date_as_key, skip this sample
                        #print_log(f"Trade date {trade_date} is not in the kline_date_as_key for {index_ticker}. Skipping this sample.", level='DEBUG')
                        skip_this_sample_window = True
                        break

                    kline_item = kline_date_as_key[trade_date]

                    if kline_item['unusual'] == True:
                        # If this date is marked as unusual, skip this sample
                        #print(f"Unusual data for {ticker} on {trade_date} in sampling window from date {trade_dates_sorted[idx]} to {trade_dates_sorted[idx + input_t_len + target_t_len - 1]}") 
                        skip_this_sample_window = True
                        break

                if skip_this_sample_window:
                    continue

                # Get the input data
                index_input_data = []
                for j in range(input_t_len):
                    trade_date = trade_dates_sorted[idx + j]
                    kline_item = index_data.klines_as_dict_ticker_as_key[index_ticker][trade_date]

                    kline_item_as_np_array = np.array([
                        kline_item['openPrice'],
                        kline_item['closePrice'],
                        kline_item['highestPrice'],
                        kline_item['lowestPrice'],
                        kline_item['turnoverVol'],
                        ], dtype=np.float32)

                    index_input_data.append(kline_item_as_np_array)

                index_input_data = np.array(index_input_data, dtype=np.float32)
                
                # Date of this sample is the last date of the input data
                last_date_of_index_input_data = trade_dates_sorted[idx + input_t_len -1]
                if last_date_of_index_input_data not in index_samples_date_as_key:
                    index_samples_date_as_key[last_date_of_index_input_data] = dict()               
                index_samples_date_as_key[last_date_of_index_input_data][index_ticker] = index_input_data


        return index_samples_date_as_key
    

    def __generate_index_samples(self, index_data, input_t_len=10, target_t_len=2):
        # Convert the klines_as_dict_ticker_as_key into training samples
        # Sort the trade dates
        trade_dates_sorted = sorted(list(index_data.klines_as_dict_date_as_key.keys()))

        index_samples_date_as_key = dict()
        for index_ticker, kline_date_as_key in index_data.klines_as_dict_ticker_as_key.items():
            print_log(f"Converting index {index_ticker} into training samples...", level='INFO')

            # Iterate through all trade dates
            for idx in range(len(trade_dates_sorted) - input_t_len - target_t_len + 1):

                skip_this_sample_window = False
                # Check if there are any unusual dates in the sample window
                for k in range(input_t_len + target_t_len):
                    trade_date = trade_dates_sorted[idx + k]
                    if trade_date not in kline_date_as_key:
                        # If the trade date is not in the kline_date_as_key, skip this sample
                        #print_log(f"Trade date {trade_date} is not in the kline_date_as_key for {index_ticker}. Skipping this sample.", level='DEBUG')
                        skip_this_sample_window = True
                        break

                    kline_item = kline_date_as_key[trade_date]

                    if kline_item['unusual'] == True:
                        # If this date is marked as unusual, skip this sample
                        #print(f"Unusual data for {ticker} on {trade_date} in sampling window from date {trade_dates_sorted[idx]} to {trade_dates_sorted[idx + input_t_len + target_t_len - 1]}") 
                        skip_this_sample_window = True
                        break

                if skip_this_sample_window:
                    continue

                # Get the input data
                index_input_data = []
                index_target_data = []

                for j in range(input_t_len):
                    trade_date = trade_dates_sorted[idx + j]
                    kline_item = index_data.klines_as_dict_ticker_as_key[index_ticker][trade_date]

                    kline_item_as_np_array = np.array([
                        kline_item['openPrice'],
                        kline_item['closePrice'],
                        kline_item['highestPrice'],
                        kline_item['lowestPrice'],
                        kline_item['turnoverVol'],
                        ], dtype=np.float32)

                    index_input_data.append(kline_item_as_np_array)

                for j in range(target_t_len):
                    trade_date = trade_dates_sorted[idx + input_t_len + j]
                    kline_item = kline_date_as_key[trade_date]

                    kline_item_as_np_array = np.array([
                        kline_item['openPrice'],
                        kline_item['closePrice'],
                        kline_item['highestPrice'],
                        kline_item['lowestPrice'],
                        kline_item['turnoverVol'],
                        ], dtype=np.float32)

                    index_target_data.append(kline_item_as_np_array)

                index_input_data = np.array(index_input_data, dtype=np.float32)
                index_target_data = np.array(index_target_data, dtype=np.float32)

                sample = {
                        'ticker': index_ticker,
                        'trade_date': trade_dates_sorted[idx + input_t_len - 1],
                        'index_input': index_input_data,
                        'index_target': index_target_data,
                    }

                # Date of this sample is the last date of the input data
                last_date_of_index_input_data = trade_dates_sorted[idx + input_t_len -1]
                if last_date_of_index_input_data not in index_samples_date_as_key:
                    index_samples_date_as_key[last_date_of_index_input_data] = dict()               
                index_samples_date_as_key[last_date_of_index_input_data][index_ticker] = sample


        return index_samples_date_as_key
    

    def __fix_stock_sample_due_to_XD_XR_DR_R_etc(self, ticker, trade_dates_sorted, kline_date_as_key, idx, stock_input_data, stock_target_data):
        """ Fix the stock input data and stock target data due to XD/XR/DR/R events."""

        stock_data_together = np.concatenate((stock_input_data, stock_target_data), axis=0)
        window_length = stock_data_together.shape[0]

        for j in range(window_length):
            trade_date = trade_dates_sorted[idx + j]
            kline_item = kline_date_as_key[trade_date]

            if 'preClosePrice' in kline_item and 'actPreClosePrice' in kline_item:
                prev_close_price = kline_item['preClosePrice']
                prev_act_close_price = kline_item['actPreClosePrice']

                if prev_close_price != prev_act_close_price:
                    # If the previous close price is not equal to the previous actual close price, 
                    # it means that there is a XD/XR/DR/R event on this date.
                    # We need to fix the stock input data and stock target data accordingly.
                    print_log(f"Fixing stock sample for {ticker} {kline_item['tickerName']} on {trade_date} due to XD/XR/DR/R event.", level='DEBUG')
                    
                    # Adjust the stock input data and stock target data
                    # Fix prices
                    stock_data_together[j:, :-1] *= (prev_act_close_price / (prev_close_price + 1e-6))
                    # Fix turnover volume, 
                    stock_data_together[j:, -1] *= (prev_close_price / (prev_act_close_price + 1e-6))

        fixed_stock_input_data = stock_data_together[:stock_input_data.shape[0]]
        fixed_stock_target_data = stock_data_together[stock_input_data.shape[0]:]

        return fixed_stock_input_data, fixed_stock_target_data

    

    def __generate_stock_samples(
            self, stock_data, input_t_len=10, target_t_len=2):
        
        # Convert the klines_as_dict_ticker_as_key into training samples
        # Sort the trade dates
        trade_dates_sorted = sorted(list(stock_data.klines_as_dict_date_as_key.keys()))

        stock_samples = []
        stock_samples_date_as_key = dict()
        for ticker, kline_date_as_key in stock_data.klines_as_dict_ticker_as_key.items():

            print_log(f"Converting stock {ticker} into training samples...", level='INFO')

            # Iterate through all trade dates
            for idx in range(len(trade_dates_sorted) - input_t_len - target_t_len + 1):

                skip_this_sample_window = False
                # Check if there are any unusual dates in the sample window
                for k in range(input_t_len + target_t_len):
                    trade_date = trade_dates_sorted[idx + k]
                    if trade_date not in kline_date_as_key:
                        # If the trade date is not in the kline_date_as_key, skip this sample
                        #print_log(f"Trade date {trade_date} is not in the kline_date_as_key for {index_ticker}. Skipping this sample.", level='DEBUG')
                        skip_this_sample_window = True
                        break

                    kline_item = kline_date_as_key[trade_date]

                    if kline_item['unusual'] == True:
                        # If this date is marked as unusual, skip this sample
                        #print(f"Unusual data for {ticker} on {trade_date} in sampling window from date {trade_dates_sorted[idx]} to {trade_dates_sorted[idx + input_t_len + target_t_len - 1]}") 
                        skip_this_sample_window = True
                        break

                if skip_this_sample_window:
                    continue
 
                # Get the input and target data
                stock_input_data = []
                stock_target_data = []

                for j in range(input_t_len):
                    trade_date = trade_dates_sorted[idx + j]
                    kline_item = kline_date_as_key[trade_date]

                    kline_item_as_np_array = np.array([
                        kline_item['openPrice'],
                        kline_item['closePrice'],
                        kline_item['highestPrice'],
                        kline_item['lowestPrice'],
                        kline_item['turnoverVol'],
                        ], dtype=np.float32)

                    stock_input_data.append(kline_item_as_np_array)

                for j in range(target_t_len):
                    trade_date = trade_dates_sorted[idx + input_t_len + j]
                    kline_item = kline_date_as_key[trade_date]

                    kline_item_as_np_array = np.array([
                        kline_item['openPrice'],
                        kline_item['closePrice'],
                        kline_item['highestPrice'],
                        kline_item['lowestPrice'],
                        kline_item['turnoverVol'],
                        ], dtype=np.float32)

                    stock_target_data.append(kline_item_as_np_array)

                stock_input_data = np.array(stock_input_data, dtype=np.float32)
                stock_target_data = np.array(stock_target_data, dtype=np.float32)

                # Fix the stock input data and stock target data due to XD/XR/DR/R events
                stock_input_data, stock_target_data = \
                    self.__fix_stock_sample_due_to_XD_XR_DR_R_etc(
                        ticker, trade_dates_sorted, kline_date_as_key, idx, stock_input_data, stock_target_data)

                sample = {
                        'ticker': ticker,
                        'trade_date': trade_dates_sorted[idx + input_t_len - 1],
                        'stock_input': stock_input_data,
                        'stock_target': stock_target_data,
                    }

                # Check if the indices sample is available for this stock sample
                if not self.__indices_sample_available_for_stock_sample(sample):
                    print_log(f"Indices sample is not available for stock sample of {ticker} on {sample['trade_date']}. Skipping this sample.", level='INFO')
                    continue

                stock_samples.append(sample)

                # Date of this sample is the last date of the input data
                last_date_of_stock_input_data = trade_dates_sorted[idx + input_t_len -1]
                if last_date_of_stock_input_data not in stock_samples_date_as_key:
                    stock_samples_date_as_key[last_date_of_stock_input_data] = dict()               
                stock_samples_date_as_key[last_date_of_stock_input_data][ticker] = sample

        return stock_samples, stock_samples_date_as_key
    

    def __plot_stock_sample_for_debug(self, input_array, target_array):
        full_array = np.concatenate((input_array, target_array), axis=0)
        # Plot this array
        plt.plot(full_array[:,0], label = "Opening Prices")
        plt.plot(full_array[:,1], label = "Closing Prices")
        plt.plot(full_array[:,2], label = "Heighest Prices")
        plt.plot(full_array[:,3], label = "Lowest Prices")
        plt.plot(full_array[:,4], label = "Turnover Volumes")
        plt.legend()
        plt.show()


    def __plot_index_sample_for_debug(self, input_array):
        # Plot this array
        for i in range(input_array.shape[1] // 5):
            plt.plot(input_array[:, i * 5], label = f"Opening Prices {i}")
            plt.plot(input_array[:, i * 5 + 1], label = f"Closing Prices {i}")
            plt.plot(input_array[:, i * 5 + 2], label = f"Heighest Prices {i}")
            plt.plot(input_array[:, i * 5 + 3], label = f"Lowest Prices {i}")
            plt.plot(input_array[:, i * 5 + 4], label = f"Turnover Volumes {i}")
        plt.legend()
        plt.show()


    def __len__(self):
        # Return the size of the dataset
        return len(self.stock_samples)
    

    def __go_through_index_klines_for_debug(self, trade_date, index_ticker, history_len=50, future_len=8):
        # Go through the index klines and print the trade dates
        if trade_date not in self.index_data.klines_as_dict_date_as_key:
            print_log(f"Trade date {trade_date} is not in the index data.", level='INFO')
            return
        
        all_trade_dates = sorted(list(self.index_data.klines_as_dict_date_as_key.keys()))
        trade_date_idx = all_trade_dates.index(trade_date)

        for i in range(trade_date_idx - history_len + 1, trade_date_idx + future_len):
            if i < 0 or i >= len(all_trade_dates):
                continue
            current_trade_date = all_trade_dates[i]
            klines_ticker_as_key = self.index_data.klines_as_dict_date_as_key[current_trade_date]
            if index_ticker not in klines_ticker_as_key:
                print_log(f"Index ticker {index_ticker} is not in the klines of trade date {current_trade_date}.", level='INFO')
                continue
            kline_item = klines_ticker_as_key[index_ticker]
            if kline_item['unusual'] == True:
                print_log(f"Trade date {current_trade_date} is marked as unusual for index ticker {index_ticker}.", level='INFO')
                continue

  
    def __indices_sample_available_for_stock_sample(self, stock_sample):
        # All index tickers
        index_tickers = sorted(list(self.index_data.klines_as_dict_ticker_as_key.keys()))
        trade_date = stock_sample['trade_date']
        
        # Check if the trade date is in the index samples
        # If not, return a zero array
        if trade_date not in self.index_samples_date_as_key:
            print_log(f"Trade date {trade_date} does not support samples of any index tickers.", level='INFO')
            #self.__go_through_index_klines_for_debug(trade_date, index_tickers[0])
            return False

        for index_ticker in index_tickers:

            if index_ticker not in self.index_samples_date_as_key[trade_date]:
                # If this index ticker is not in the index samples, return a zero array
                print_log(f"Trade date {trade_date} does not support a sample of index ticker {index_ticker}.", level='INFO')
                #self.__go_through_index_klines_for_debug(trade_date, index_ticker)
                return False

        return True
    

    def __indices_sample_via_stock_sample(self, stock_sample):
        # Combine the index input sample and stock input sample

        # All index tickers
        index_tickers = sorted(list(self.index_data.klines_as_dict_ticker_as_key.keys()))
        trade_date = stock_sample['trade_date']
        
        # Check if the trade date is in the index samples
        # If not, return a zero array
        if trade_date not in self.index_samples_date_as_key:
            print_log(f"Trade date {trade_date} does not support samples of any index tickers.", level='WARNING')
            #self.__go_through_index_klines_for_debug(trade_date, index_tickers[0])
            return np.zeros((stock_sample['stock_input'].shape[0], len(index_tickers) * 5), dtype=np.float32)

        indices_input_data = []
        indices_target_data = []
        for index_ticker in index_tickers:

            if index_ticker not in self.index_samples_date_as_key[trade_date]:
                # If this index ticker is not in the index samples, return a zero array
                print_log(f"Trade date {trade_date} does not support a sample of index ticker {index_ticker}.", level='WARNING')
                #self.__go_through_index_klines_for_debug(trade_date, index_ticker)
                index_input_data = np.zeros((stock_sample['stock_input'].shape[0], 5), dtype=np.float32)
            else:
                index_input_data = self.index_samples_date_as_key[trade_date][index_ticker]['index_input']
                index_target_data = self.index_samples_date_as_key[trade_date][index_ticker]['index_target']
            indices_input_data.append(index_input_data)
            indices_target_data.append(index_target_data)
        
        indices_input_data = np.concatenate(indices_input_data, axis=1)
        indices_target_data = np.concatenate(indices_target_data, axis=1)

        return indices_input_data, indices_target_data
    

    def get_name_via_ticker(self, ticker, trade_date=None):
        # Get the name of the stock via ticker
        if trade_date is None:
            if ticker not in self.stock_data.names_ticker_as_key:
                return 'unkown'
            return self.stock_data.names_ticker_as_key[ticker]
        else:
            if trade_date not in self.stock_data.klines_as_dict_date_as_key:
                if ticker not in self.stock_data.names_ticker_as_key:
                    return 'unkown'
                return self.stock_data.names_ticker_as_key[ticker]
            else:
                klines_ticker_as_key = self.stock_data.klines_as_dict_date_as_key[trade_date]
                if ticker not in klines_ticker_as_key:
                    if ticker not in self.stock_data.names_ticker_as_key:
                        return 'unkown'
                    return self.stock_data.names_ticker_as_key[ticker]
                else:
                    kline_item = klines_ticker_as_key[ticker]
                    if 'tickerName' not in kline_item:
                        if ticker not in self.stock_data.names_ticker_as_key:
                            return 'unkown'
                        return self.stock_data.names_ticker_as_key[ticker]
                    else:
                        return kline_item['tickerName']
    

    def get_original_sample(self, idx):
        # Retrieve the data sample at index `idx`
        stock_sample = self.stock_samples[idx]

        indices_input_array, indices_target_array = self.__indices_sample_via_stock_sample(stock_sample)
        stock_input_array = stock_sample['stock_input']
        stock_target_array = stock_sample['stock_target']

        return indices_input_array, indices_target_array, stock_input_array, stock_target_array
    

    def get_original_sample_by_date_and_ticker(self, date, ticker):
        if date not in self.stock_samples_date_as_key:
            raise ValueError(f"Date {date} not found in the dataset.")
        if ticker not in self.stock_samples_date_as_key[date]:
            raise ValueError(f"Ticker {ticker} not found for date {date} in the dataset.")
        stock_sample = self.stock_samples_date_as_key[date][ticker]

        indices_input_array, indices_target_array = self.__indices_sample_via_stock_sample(stock_sample)
        stock_input_array = stock_sample['stock_input']
        stock_target_array = stock_sample['stock_target']

        return indices_input_array, indices_target_array, stock_input_array, stock_target_array
    

    def get_original_samples_by_date(self, date):
        if date not in self.stock_samples_date_as_key:
            raise ValueError(f"Date {date} not found in the dataset.")
        stock_samples = self.stock_samples_date_as_key[date]

        original_samples_ticker_as_key = dict()
        for ticker, stock_sample in stock_samples.items():
            indices_input_array, indices_target_array = self.__indices_sample_via_stock_sample(stock_sample)
            stock_input_array = stock_sample['stock_input']
            stock_target_array = stock_sample['stock_target']

            original_sample = (indices_input_array, indices_target_array, stock_input_array, stock_target_array)
            original_samples_ticker_as_key[ticker] = original_sample
            
        return original_samples_ticker_as_key
    

    def get_all_tickers_sample_available_on_date(self, date):
        if date not in self.stock_samples_date_as_key:
            return set()
        return set(self.stock_samples_date_as_key[date].keys())
    

    def get_all_tickers_of_date(self, date):
        """
        Get all tickers of the date, including those whose training sample is available on this date.
        """
        if date not in self.stock_data.klines_as_dict_date_as_key:
            raise set()
        return set(self.stock_data.klines_as_dict_date_as_key[date].keys())
    

    def get_label(self, idx):
        # Retrieve the data sample at index `idx`
        indices_input_array, indices_target_array, stock_input_array, stock_target_array = self.get_original_sample(idx)
        # Classify the target array using the classification rule
        cls_label, rate_label = self.cls_rule.classify(stock_input_array, stock_target_array)

        return cls_label, rate_label


    def __getitem__(self, idx):

        indices_input_array, indices_target_array, stock_input_array, stock_target_array = self.get_original_sample(idx)
        # Classify the target array using the classification rule
        cls_label, rate_label = self.cls_rule.classify(stock_input_array, stock_target_array)

        # Return the original input and target tensors
        return indices_input_array, indices_target_array, stock_input_array, stock_target_array, cls_label, rate_label
        

    def __auto_determine_cls_rule_old(self, n_classes=5):
        """
        Automatically determine the classification rule based on the data.
        This method should be called before using the dataset.
        """
        assert n_classes % 2 == 1, "n_classes must be an odd number."
        price_change_rate_list = []
        for i in range(len(self.stock_samples)):
            indices_input_array, indices_target_array, stock_input_array, stock_target_array = self.get_original_sample(i)
            price_change_rate = self.cls_rule.price_change_rate(stock_input_array, stock_target_array)
            price_change_rate_list.append(price_change_rate)

        n_samples_for_each_cls = len(price_change_rate_list) // n_classes
        price_change_rate_list = sorted(price_change_rate_list)
        cls_threshold_list = []
        for i in range(n_classes - 1):
            cls_threshold_list.append(price_change_rate_list[(i + 1) * n_samples_for_each_cls])

        return cls_threshold_list



    def __auto_determine_cls_rule(self, n_classes=5):
        """
        Automatically determine the classification rule based on the data.
        This method should be called before using the dataset.
        """
        assert n_classes % 2 == 1, "n_classes must be an odd number."
        # Get all the cls_labels from the samples
        price_change_rate_list = []
        for i in range(len(self.stock_samples)):
            indices_input_array, indices_target_array, stock_input_array, stock_target_array = self.get_original_sample(i)
            price_change_rate = self.cls_rule.price_change_rate(stock_input_array, stock_target_array)
            price_change_rate_list.append(price_change_rate)

        price_change_rate_list = sorted(price_change_rate_list)

        # Search for the place where price change rate is turning from negative to positive
        idx_range = (0, len(price_change_rate_list) - 1)
        mid_idx = 0
        while idx_range[1] - idx_range[0] > 1:
            mid_idx = (idx_range[0] + idx_range[1]) // 2
            if price_change_rate_list[mid_idx] < 0:
                idx_range = (mid_idx, idx_range[1])
            else:
                idx_range = (idx_range[0], mid_idx)

        # Fillin negative and positive price change rate list
        # The first half of the list is negative and the second half is positive

        rate_list_neg = price_change_rate_list[:idx_range[0]]
        rate_list_pos = price_change_rate_list[idx_range[0]:]

        n_samples_for_each_cls_half_neg = len(rate_list_neg) // n_classes
        n_samples_for_each_cls_half_pos = len(rate_list_pos) // n_classes

        cls_threshold_list_neg = []
        for i in range((n_classes - 1) // 2):
            cls_threshold_list_neg.append(price_change_rate_list[(i + 1) * n_samples_for_each_cls_half_neg * 2])
        cls_threshold_list_pos = []
        for i in range((n_classes - 1) // 2):
            cls_threshold_list_pos.append(price_change_rate_list[-(i + 1) * n_samples_for_each_cls_half_pos * 2])
        cls_threshold_list_pos = cls_threshold_list_pos[::-1]
        cls_threshold_list = cls_threshold_list_neg + cls_threshold_list_pos

        # Amend the threshold list to make it symmetric
        cls_threshold_list_symmetric = [0.0] * (n_classes - 1)
        for i in range((n_classes - 1) // 2):
            cls_threshold_list_symmetric[i] = -(- cls_threshold_list_neg[i] + cls_threshold_list_pos[-1-i]) / 2
            cls_threshold_list_symmetric[-1-i] = - cls_threshold_list_symmetric[i]

        thresholds_valid = True
        for i in range(1, len(cls_threshold_list_symmetric)):
            if cls_threshold_list_symmetric[i] <= cls_threshold_list_symmetric[i - 1]:
                thresholds_valid = False
                break
        if not thresholds_valid:
            raise ValueError("Thresholds must be in ascending order.")

        return cls_threshold_list_symmetric


    def merge(self, other_dataset):
        """
        Merge another dataset into this dataset.
        """
        if not isinstance(other_dataset, ClsDataset):
            raise ValueError("The other dataset must be an instance of ClsDataset.")

        # Merge stock data
        self.stock_data.klines_as_dict_date_as_key.update(other_dataset.stock_data.klines_as_dict_date_as_key)
        for ticker, kline_date_as_key in other_dataset.stock_data.klines_as_dict_ticker_as_key.items():
            if ticker not in self.stock_data.klines_as_dict_ticker_as_key:
                self.stock_data.klines_as_dict_ticker_as_key[ticker] = dict()
            # Update the ticker data
            self.stock_data.klines_as_dict_ticker_as_key[ticker].update(kline_date_as_key)
        self.stock_data.names_ticker_as_key.update(other_dataset.stock_data.names_ticker_as_key)

        # Merge index data
        self.index_data.klines_as_dict_date_as_key.update(other_dataset.index_data.klines_as_dict_date_as_key)
        for ticker, kline_date_as_key in other_dataset.index_data.klines_as_dict_ticker_as_key.items():
            if ticker not in self.index_data.klines_as_dict_ticker_as_key:
                self.index_data.klines_as_dict_ticker_as_key[ticker] = dict()
            # Update the ticker data
            self.index_data.klines_as_dict_ticker_as_key[ticker].update(kline_date_as_key)
        self.index_data.names_ticker_as_key.update(other_dataset.index_data.names_ticker_as_key)

        # Merge samples
        self.index_samples_date_as_key.update(other_dataset.index_samples_date_as_key)
        self.stock_samples_date_as_key.update(other_dataset.stock_samples_date_as_key)
        self.stock_samples.extend(other_dataset.stock_samples)

        # Re-generate cls_rule
        n_classes = len(self.cls_rule.cls_threshold_list) + 1
        print_log(f"Auto-determining the classification rule with {n_classes} classes...", level='INFO')
        cls_threshold_list = self.__auto_determine_cls_rule(n_classes=n_classes)
        self.cls_rule.cls_threshold_list = cls_threshold_list


def test_case(market, stock_root_dir, index_root_dir, exchange_cd, index_tickers, first_date, last_date):
    # Test case for the ClsDataset class
    #stock_root_dir = '/dir/to/my/data/ashare_daily_stock_order_by_dates'
    #index_root_dir = '/dir/to/my/data/ashare_daily_index_order_by_dates'
    start_date = first_date
    end_date = last_date

    dataset = ClsDataset(
        stock_root_dir, index_root_dir,
        market, start_date, end_date,
        exchange_cd=exchange_cd,
        stock_tickers=None,
        index_tickers=index_tickers,
        strong_decrease_ignore_threshold=-1000,
        strong_increase_ignore_threshold=1000,
        strong_fluctuation_ignore_threshold=1000,
        input_t_len=50, target_t_len=8, n_classes=7)
    
    print(f"len of data: {len(dataset)}")
    for idx in range(10):
        indices_input_array, indices_target_array, stock_input_array, stock_target_array, cls_label, rate_label = dataset[idx]
        print(f"Sample {idx}:")
        print(f"Indices input shape: {indices_input_array.shape}")
        print(f"Indices target shape: {indices_target_array.shape}")
        print(f"Stock input shape: {stock_input_array.shape}")
        print(f"Stock target shape: {stock_target_array.shape}")
        print(f"Class label: {cls_label}")
        print(f"Rate label: {rate_label}")

    print([f'{x:.4f}' for x in dataset.cls_rule.cls_threshold_list])
    
    cls_statistics = dict()
    for i in range(len(dataset)):
        cls_label, rate_label = dataset.get_label(i)
        if cls_label not in cls_statistics:
            cls_statistics[cls_label] = 1
        else:
            cls_statistics[cls_label] += 1
    print_log("Number of samples:", level='INFO')
    print_log(f"Total: {len(dataset)}", level='INFO')
    print_log("Number of samples in each class:", level='INFO')
    print_log(f"cls_statistics: {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')
    print_log("Press Enter to exit...", level='INFO')

    index_dates = sorted(dataset.index_samples_date_as_key.keys())
    stock_dates = sorted(dataset.stock_samples_date_as_key.keys())

    for stock_date in stock_dates:
        if stock_date not in index_dates:
            print_log(f"Stock date {stock_date} is not in index dates.", level='WARNING')
            continue
        #index_tickers = dataset.index_samples_date_as_key[stock_date].keys()
        #stock_tickers = dataset.stock_samples_date_as_key[stock_date].keys()
        #print_log(f"Stock date {stock_date} has {len(index_tickers)} index tickers and {len(stock_tickers)} stock tickers.", level='INFO')


def load_dataset(config):

    # Hyperparameters
    # Access the parameters
    # Time series lengths
    input_t_len = config['dataset']['sequence_length']
    output_t_len = config['dataset']['target_length']

    # Load dataset
    train_dataset = ClsDataset(
        stock_root_dir=config['dataset']['stock_root_dir'],
        index_root_dir=config['dataset']['index_root_dir'],
        market=config['dataset']['market'],
        start_date=config['dataset']['training']['start_date'],
        end_date=config['dataset']['training']['end_date'],
        exchange_cd=config['dataset']['exchange_cd'],
        stock_tickers=config['dataset']['stock_tickers'],
        stock_tickers_to_exclude=config['dataset']['stock_tickers_to_exclude'],
        index_tickers=config['dataset']['index_tickers'],
        input_t_len=input_t_len,
        target_t_len=output_t_len,
        min_daily_turnover_value=config['preprocessing']['min_daily_turnover_value'],
        strong_decrease_ignore_threshold=config['preprocessing']['strong_decrease_ignore_threshold'],
        strong_increase_ignore_threshold=config['preprocessing']['strong_increase_ignore_threshold'],
        strong_fluctuation_ignore_threshold=config['preprocessing']['strong_fluctuation_ignore_threshold'],
        n_classes=config['dataset']['n_classes'])
    
    val_dataset_1 = ClsDataset(
        stock_root_dir=config['dataset']['stock_root_dir'],
        index_root_dir=config['dataset']['index_root_dir'],
        market=config['dataset']['market'],
        start_date=config['dataset']['validation_1']['start_date'],
        end_date=config['dataset']['validation_1']['end_date'],
        exchange_cd=config['dataset']['exchange_cd'],
        stock_tickers=config['dataset']['stock_tickers'],
        stock_tickers_to_exclude=config['dataset']['stock_tickers_to_exclude'],
        index_tickers=config['dataset']['index_tickers'],
        input_t_len=input_t_len,
        target_t_len=output_t_len,
        min_daily_turnover_value=config['preprocessing']['min_daily_turnover_value'],
        strong_decrease_ignore_threshold=config['preprocessing']['strong_decrease_ignore_threshold'],
        strong_increase_ignore_threshold=config['preprocessing']['strong_increase_ignore_threshold'],
        strong_fluctuation_ignore_threshold=config['preprocessing']['strong_fluctuation_ignore_threshold'],
        cls_rule=train_dataset.cls_rule.cls_threshold_list)
    
    val_dataset_2 = ClsDataset(
        stock_root_dir=config['dataset']['stock_root_dir'],
        index_root_dir=config['dataset']['index_root_dir'],
        market=config['dataset']['market'],
        start_date=config['dataset']['validation_2']['start_date'],
        end_date=config['dataset']['validation_2']['end_date'],
        exchange_cd=config['dataset']['exchange_cd'],
        stock_tickers=config['dataset']['stock_tickers'],
        stock_tickers_to_exclude=config['dataset']['stock_tickers_to_exclude'],
        index_tickers=config['dataset']['index_tickers'],
        input_t_len=input_t_len,
        target_t_len=output_t_len,
        min_daily_turnover_value=config['preprocessing']['min_daily_turnover_value'],
        strong_decrease_ignore_threshold=config['preprocessing']['strong_decrease_ignore_threshold'],
        strong_increase_ignore_threshold=config['preprocessing']['strong_increase_ignore_threshold'],
        strong_fluctuation_ignore_threshold=config['preprocessing']['strong_fluctuation_ignore_threshold'],
        cls_rule=train_dataset.cls_rule.cls_threshold_list)
    
    print_log([f'{x:.4f}' for x in train_dataset.cls_rule.cls_threshold_list], level='INFO')                                                           
    cls_statistics = dict()
    for i in range(len(train_dataset)):
        cls_label, rate_label = train_dataset.get_label(i)
        if cls_label not in cls_statistics:
            cls_statistics[cls_label] = 1
        else:
            cls_statistics[cls_label] += 1
    print_log(f"cls_statistics for train_dataset: {len(train_dataset)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    print_log([f'{x:.4f}' for x in val_dataset_1.cls_rule.cls_threshold_list], level='INFO')
    cls_statistics = dict()
    for i in range(len(val_dataset_1)):
        cls_label, rate_label = val_dataset_1.get_label(i)
        if cls_label not in cls_statistics:
            cls_statistics[cls_label] = 1
        else:
            cls_statistics[cls_label] += 1
    print_log(f"cls_statistics for val_dataset_1: {len(val_dataset_1)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    print_log([f'{x:.4f}' for x in val_dataset_2.cls_rule.cls_threshold_list], level='INFO')
    cls_statistics = dict()
    for i in range(len(val_dataset_2)):
        cls_label, rate_label = val_dataset_2.get_label(i)
        if cls_label not in cls_statistics:
            cls_statistics[cls_label] = 1
        else:
            cls_statistics[cls_label] += 1
    print_log(f"cls_statistics for val_dataset_2: {len(val_dataset_2)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    return train_dataset, val_dataset_1, val_dataset_2


def load_dataset_via_single_range(config, start_date, end_date):
    input_t_len = config['dataset']['sequence_length']
    output_t_len = config['dataset']['target_length']

    dataset = ClsDataset(
        stock_root_dir=config['dataset']['stock_root_dir'],
        index_root_dir=config['dataset']['index_root_dir'],
        market=config['dataset']['market'],
        start_date=start_date,
        end_date=end_date,
        exchange_cd=config['dataset']['exchange_cd'],
        stock_tickers=config['dataset']['stock_tickers'],
        stock_tickers_to_exclude=config['dataset']['stock_tickers_to_exclude'],
        index_tickers=config['dataset']['index_tickers'],
        input_t_len=input_t_len,
        target_t_len=output_t_len,
        min_daily_turnover_value=config['preprocessing']['min_daily_turnover_value'],
        strong_decrease_ignore_threshold=config['preprocessing']['strong_decrease_ignore_threshold'],
        strong_increase_ignore_threshold=config['preprocessing']['strong_increase_ignore_threshold'],
        strong_fluctuation_ignore_threshold=config['preprocessing']['strong_fluctuation_ignore_threshold'],
        n_classes=config['dataset']['n_classes'])

    return dataset


def load_crossed_dataset(config):

    train_dataset_date_ranges = config['dataset']['training']['date_ranges']
    train_dataset = None
    for start_date, end_date in train_dataset_date_ranges:
        new_train_dataset = load_dataset_via_single_range(
            config, start_date, end_date)
        if train_dataset is None:
            train_dataset = new_train_dataset
        else:
            train_dataset.merge(new_train_dataset)

    val_dataset_1_date_ranges = config['dataset']['validation_1']['date_ranges']
    val_dataset_1 = None
    for start_date, end_date in val_dataset_1_date_ranges:
        new_val_dataset_1 = load_dataset_via_single_range(
            config, start_date, end_date)
        if val_dataset_1 is None:
            val_dataset_1 = new_val_dataset_1
        else:
            val_dataset_1.merge(new_val_dataset_1)

    val_dataset_2_date_ranges = config['dataset']['validation_2']['date_ranges']
    val_dataset_2 = None
    for start_date, end_date in val_dataset_2_date_ranges:
        new_val_dataset_2 = load_dataset_via_single_range(
            config, start_date, end_date)
        if val_dataset_2 is None:
            val_dataset_2 = new_val_dataset_2
        else:
            val_dataset_2.merge(new_val_dataset_2)

    ## All datasets should share the same classification rule
    if val_dataset_1 is not None:
        val_dataset_1.cls_rule = train_dataset.cls_rule
    if val_dataset_2 is not None:
        val_dataset_2.cls_rule = train_dataset.cls_rule

    print_log([f'{x:.4f}' for x in train_dataset.cls_rule.cls_threshold_list], level='INFO')                                                           
    cls_statistics = dict()
    for i in range(len(train_dataset)):
        cls_label, rate_label = train_dataset.get_label(i)
        if cls_label not in cls_statistics:
            cls_statistics[cls_label] = 1
        else:
            cls_statistics[cls_label] += 1
    print_log(f"cls_statistics for train_dataset: {len(train_dataset)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    if val_dataset_1 is not None:
        print_log([f'{x:.4f}' for x in val_dataset_1.cls_rule.cls_threshold_list], level='INFO')
        cls_statistics = dict()
        for i in range(len(val_dataset_1)):
            cls_label, rate_label = val_dataset_1.get_label(i)
            if cls_label not in cls_statistics:
                cls_statistics[cls_label] = 1
            else:
                cls_statistics[cls_label] += 1
        print_log(f"cls_statistics for val_dataset_1: {len(val_dataset_1)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    if val_dataset_2 is not None:
        print_log([f'{x:.4f}' for x in val_dataset_2.cls_rule.cls_threshold_list], level='INFO')
        cls_statistics = dict()
        for i in range(len(val_dataset_2)):
            cls_label, rate_label = val_dataset_2.get_label(i)
            if cls_label not in cls_statistics:
                cls_statistics[cls_label] = 1
            else:
                cls_statistics[cls_label] += 1
        print_log(f"cls_statistics for val_dataset_2: {len(val_dataset_2)} {sorted(cls_statistics.items(), key=lambda x: x[0])}", level='INFO')

    return train_dataset, val_dataset_1, val_dataset_2



if __name__ == "__main__":

    # arg parse
    import argparse
    parser = argparse.ArgumentParser(description='Test ClsDataset')
    parser.add_argument('--first_date', type=str, default='1990-01-01', help='First date of the dataset')
    parser.add_argument('--last_date', type=str, default='2099-12-31', help='Last date of the dataset')
    args = parser.parse_args()


    configure_logger(
        log_name='cls_dataset',
        config = {
            'logging': {
                'logging_level': 'INFO',
                'log_dir': '/dir/to/my/logs/',
            }
        },
        log_to_file=True,
    )

    # Test the ClsDataset class

    test_case(
        market = 'ashare',
        stock_root_dir = '/dir/to/my/data/ashare_daily_stock_order_by_dates',
        index_root_dir = '/dir/to/my/data/ashare_daily_index_order_by_dates',
        exchange_cd = '',
        index_tickers = ['000001', '399001', '399300'],
        first_date=args.first_date,
        last_date=args.last_date)

    '''
    test_case(
        market= 'hk',
        stock_root_dir = '/dir/to/my/data/hk_daily_stock_order_by_dates',
        index_root_dir = '/dir/to/my/data/hk_daily_index_order_by_dates',
        exchange_cd= 'XHKG',
        index_tickers = ['HSI', 'HSCCI'])
    '''