import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from train_daily_data.daily_data import StockData
from train_daily_data.global_logger import print_log, configure_logger

class InferDataset():

    def __init__(self, 
                stock_root_dir,
                index_root_dir,
                market : str, 
                start_date : str, 
                end_date : str,
                exchange_cd : str=None, 
                stock_tickers : Union[set, list, tuple]=None,
                stock_tickers_to_exclude : Union[set, list, tuple]=None,
                index_tickers : Union[set, list, tuple]=None,
                index_tickers_to_exclude : Union[set, list, tuple]=None,
                min_daily_turnover_value=0.0,
                strong_decrease_ignore_threshold=-0.2,
                strong_increase_ignore_threshold=0.25,
                strong_fluctuation_ignore_threshold=0.3,
                input_t_len=10) -> None:

        super(InferDataset, self).__init__()


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
            min_daily_turnover_value=min_daily_turnover_value,
            strong_decrease_ignore_threshold=strong_decrease_ignore_threshold,
            strong_increase_ignore_threshold=strong_increase_ignore_threshold,
            strong_fluctuation_ignore_threshold=strong_fluctuation_ignore_threshold)

        self.__fix_trade_dates(self.index_data, self.stock_data)

        self.index_samples_date_as_key = self.__generate_index_samples(
            self.index_data,
            input_t_len=input_t_len)

        self.stock_samples_date_as_key = self.__generate_stock_samples(
            self.stock_data,
            input_t_len=input_t_len)
        
        self.sample_print = False


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
    

    def __generate_index_samples(self, index_data, input_t_len=10):
        # Convert the klines_as_dict_ticker_as_key into training samples
        # Sort the trade dates
        trade_dates_sorted = sorted(list(index_data.klines_as_dict_date_as_key.keys()))

        index_samples_date_as_key = dict()
        for index_ticker, kline_date_as_key in index_data.klines_as_dict_ticker_as_key.items():
            print_log(f"Converting {index_ticker} into training samples...", level='DEBUG')

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
    

    def __fix_stock_sample_due_to_XD_XR_DR_R_etc(self, ticker, trade_dates_sorted, kline_date_as_key, idx, stock_input_data):
        """ Fix the stock input data and stock target data due to XD/XR/DR/R events."""

        window_length = stock_input_data.shape[0]
        fixed_stock_input_data = np.copy(stock_input_data)

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
                    fixed_stock_input_data[j:, :-1] *= (prev_act_close_price / prev_close_price)
                    # Fix turnover volume, 
                    fixed_stock_input_data[j:, -1] *= (prev_close_price / prev_act_close_price)

        return fixed_stock_input_data
    

    def __generate_stock_samples(self, stock_data, input_t_len=10):
        
        # Convert the klines_as_dict_ticker_as_key into training samples
        # Sort the trade dates
        trade_dates_sorted = sorted(list(stock_data.klines_as_dict_date_as_key.keys()))

        stock_samples_date_as_key = dict()
        for ticker, kline_date_as_key in stock_data.klines_as_dict_ticker_as_key.items():

            print_log(f"Converting {ticker} into training samples...", level='DEBUG')

            # Iterate through all trade dates
            for idx in range(len(trade_dates_sorted) - input_t_len + 1):

                skip_this_sample_window = False
                # Check if there are any unusual dates in the sample window
                for k in range(input_t_len):
                    trade_date = trade_dates_sorted[idx + k]
                    if trade_date not in kline_date_as_key:
                        # If the trade date is not in the kline_date_as_key, skip this sample
                        #print_log(f"Trade date {trade_date} is not in the kline_date_as_key for {ticker}. Skipping this sample.", level='DEBUG')
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

                stock_input_data = np.array(stock_input_data, dtype=np.float32) 
                # If there is a XD/XR/DR/R event, we need to fix the stock input data and stock target data
                stock_input_data = self.__fix_stock_sample_due_to_XD_XR_DR_R_etc(
                    ticker, trade_dates_sorted, kline_date_as_key, idx, stock_input_data)              

                sample = {
                        'ticker': ticker,
                        'trade_date': trade_dates_sorted[idx + input_t_len - 1],
                        'stock_input': stock_input_data,
                    }
                
                # Date of this sample is the last date of the input data
                last_date_of_stock_input_data = trade_dates_sorted[idx + input_t_len -1]
                if last_date_of_stock_input_data not in stock_samples_date_as_key:
                    stock_samples_date_as_key[last_date_of_stock_input_data] = dict()               
                stock_samples_date_as_key[last_date_of_stock_input_data][ticker] = sample

        return stock_samples_date_as_key
    

    def plot_stock_sample_for_debug(self, input_array):

        # Plot this array
        plt.plot(input_array[:,0], label = "Opening Prices")
        plt.plot(input_array[:,1], label = "Closing Prices")
        plt.plot(input_array[:,2], label = "Heighest Prices")
        plt.plot(input_array[:,3], label = "Lowest Prices")
        #plt.plot(input_array[:,4], label = "Turnover Volumes")
        plt.legend()
        plt.show()


    def plot_index_sample_for_debug(self, input_array):
        # Plot this array
        for i in range(input_array.shape[1] // 5):
            plt.plot(input_array[:, i * 5], label = f"Opening Prices {i}")
            plt.plot(input_array[:, i * 5 + 1], label = f"Closing Prices {i}")
            plt.plot(input_array[:, i * 5 + 2], label = f"Heighest Prices {i}")
            plt.plot(input_array[:, i * 5 + 3], label = f"Lowest Prices {i}")
            #plt.plot(input_array[:, i * 5 + 4], label = f"Turnover Volumes {i}")
        plt.legend()
        plt.show()


    def __len__(self):

        count = 0
        for date, stock_samples_ticker_as_key in self.stock_samples_date_as_key.items():
            count += len(stock_samples_ticker_as_key)
        # Return the size of the dataset
        return count
        

    def __indices_sample_via_stock_sample(self, stock_sample):
        # Combine the index input sample and stock input sample

        # All index tickers
        index_tickers = sorted(list(self.index_data.klines_as_dict_ticker_as_key.keys()))
        trade_date = stock_sample['trade_date']
        
        # Check if the trade date is in the index samples
        # If not, return a zero array
        if trade_date not in self.index_samples_date_as_key:
            print_log(f"Trade date {trade_date} does not support any index tickers needed.", level='DEBUG')
            return np.zeros((stock_sample['stock_input'].shape[0], len(index_tickers) * 5), dtype=np.float32)

        indices_input_data = []
        for index_ticker in index_tickers:

            if index_ticker not in self.index_samples_date_as_key[trade_date]:
                # If this index ticker is not in the index samples, return a zero array
                print_log(f"Trade date {trade_date} does not support index ticker {index_ticker}.", level='DEBUG')
                index_input_data = np.zeros((stock_sample['stock_input'].shape[0], 5), dtype=np.float32)
            else:
                index_input_data = self.index_samples_date_as_key[trade_date][index_ticker]
            indices_input_data.append(index_input_data)
        
        indices_input_data = np.concatenate(indices_input_data, axis=1)

        return indices_input_data


    def get_name_via_ticker(self, ticker, trade_date=None):
        # Get the name of the stock via ticker
        if trade_date is None:
            if ticker not in self.stock_data.names_ticker_as_key:
                return 'unknown'
            return self.stock_data.names_ticker_as_key[ticker]
        else:
            if trade_date not in self.stock_data.klines_as_dict_date_as_key:
                if ticker not in self.stock_data.names_ticker_as_key:
                    return 'unknown'
                return self.stock_data.names_ticker_as_key[ticker]
            else:
                klines_ticker_as_key = self.stock_data.klines_as_dict_date_as_key[trade_date]
                if ticker not in klines_ticker_as_key:
                    if ticker not in self.stock_data.names_ticker_as_key:
                        return 'unknown'
                    return self.stock_data.names_ticker_as_key[ticker]
                else:
                    kline_item = klines_ticker_as_key[ticker]
                    if 'tickerName' not in kline_item:
                        if ticker not in self.stock_data.names_ticker_as_key:
                            return 'unknown'
                        return self.stock_data.names_ticker_as_key[ticker]
                    else:
                        return kline_item['tickerName']
    

    def get_original_sample_by_date_and_ticker(self, date, ticker):
        if date not in self.stock_samples_date_as_key:
            raise ValueError(f"Date {date} not found in the dataset.")
        if ticker not in self.stock_samples_date_as_key[date]:
            raise ValueError(f"Ticker {ticker} not found for date {date} in the dataset.")
        stock_sample = self.stock_samples_date_as_key[date][ticker]

        indices_input_array = self.__indices_sample_via_stock_sample(stock_sample)
        stock_input_array = stock_sample['stock_input']

        return indices_input_array, stock_input_array
    

    def get_original_samples_by_date(self, date):
        if date not in self.stock_samples_date_as_key:
            raise ValueError(f"Date {date} not found in the dataset.")
        stock_samples = self.stock_samples_date_as_key[date]

        original_samples_ticker_as_key = dict()
        for ticker, stock_sample in stock_samples.items():
            indices_input_array = self.__indices_sample_via_stock_sample(stock_sample)
            stock_input_array = stock_sample['stock_input']

            original_sample = (indices_input_array, stock_input_array)
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
            return set()
        return set(self.stock_data.klines_as_dict_date_as_key[date].keys())


    def get_all_trade_dates(self):
        return sorted(list(self.stock_samples_date_as_key.keys()))


def test_case(differenced=False):
    # Test case for the ClsDataset class
    stock_root_dir = 'datayes_data_sample/ashare_daily_stock_order_by_dates'
    index_root_dir = 'datayes_data_sample/ashare_daily_index_order_by_dates'
    market = 'ashare'
    start_date = '2025-01-01'
    end_date = '2025-06-11'
    ticker = None

    dataset = InferDataset(
        stock_root_dir, index_root_dir,
        market, start_date, end_date,
        exchange_cd='XSHE',
        stock_tickers=None,
        index_tickers=['399001',],
        max_tradedate_gap=0,
        input_t_len=20)
    
    print(f"len of data: {len(dataset)}")

    indices_input_array, stock_input_array = dataset.get_original_sample_by_date_and_ticker('2025-06-11', '300394')
    
    # plot indices_input_array and stock_input_array
    dataset.plot_index_sample_for_debug(indices_input_array)

    dataset.plot_stock_sample_for_debug(stock_input_array)



if __name__ == "__main__":

    configure_logger(
        log_file_name='none',
        config = {
            'logging': {
                'logging_level': 'DEBUG',
                'log_dir': '/dir/to/my/logs/',
            }
        },
        log_to_file=False,
    )

    # Test the ClsDataset class
    test_case()
    
    #test_case(differenced=True)