from train_daily_data.infer_dataset import InferDataset
from train_daily_data.global_logger import print_log
from decimal import Decimal, ROUND_HALF_UP, ROUND_UP, getcontext

import numpy as np

def load_infer_dataset(config, final_test=True):

    if final_test:
        start_date = config['dataset']['final_testing']['start_date']
        end_date = config['dataset']['final_testing']['end_date']
    else:
        start_date = config['dataset']['testing']['start_date']
        end_date = config['dataset']['testing']['end_date']

    infer_dataset = InferDataset(
        stock_root_dir=config['dataset']['stock_root_dir'],
        index_root_dir=config['dataset']['index_root_dir'],
        market=config['dataset']['market'],
        start_date=start_date,
        end_date=end_date,
        exchange_cd=config['dataset']['exchange_cd'],
        stock_tickers=config['dataset']['stock_tickers'],
        stock_tickers_to_exclude=config['dataset']['stock_tickers_to_exclude'],
        index_tickers=config['dataset']['index_tickers'],
        input_t_len=config['dataset']['sequence_length'],
        min_daily_turnover_value=0.0,
        strong_decrease_ignore_threshold=-10000.0,
        strong_increase_ignore_threshold=10000.0,
        strong_fluctuation_ignore_threshold=10000.0)
    
    return infer_dataset

class CurrencyReservoir:
    def __init__(self, initial_amount : Decimal = Decimal(0.00)):
        if not isinstance(initial_amount, Decimal):
            raise ValueError("Initial amount must be a Decimal.")
        if initial_amount < 0:
            raise ValueError("Initial amount must be a positive Decimal.")
        self.amount = initial_amount

    def save(self, amount : Decimal):
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be a Decimal.")
        if amount < 0:
            raise ValueError("Amount must be a positive Decimal.")
        self.amount += amount

    def withdraw(self, amount : Decimal):
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be a Decimal.")
        if amount < 0:
            raise ValueError("Amount must be a positive Decimal.")
        
        if self.amount < amount:
            return Decimal(0.00)

        self.amount -= amount
        return amount

    def get_amount(self):
        return self.amount
    
    def force_set_amount(self, amount : Decimal):
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be a Decimal.")
        if amount < 0:
            raise ValueError("Amount must be a positive Decimal.")
        self.amount = amount

class TickerReservoir:
    def __init__(self, initial_volume : int = 0):
        if not isinstance(initial_volume, int):
            raise ValueError("Initial volume must be an integer.")
        if initial_volume < 0:
            raise ValueError("Volume must be a positive integer.")
        self.volume = initial_volume

    def save(self, volume):
        if not isinstance(volume, int):
            raise ValueError("Volume must be an integer.")
        if volume < 0:
            raise ValueError("Volume must be a positive integer.")

        self.volume += volume

    def withdraw(self, volume):
        if not isinstance(volume, int):
            raise ValueError("Amount must be an integer.")
        if volume < 0:
            raise ValueError("Amount must be a positive integer.")
        if self.volume < volume:
            return 0

        self.volume -= volume
        return volume

    def get_volume(self):
        return self.volume
    
    def amend_volume(self, volume):
        if not isinstance(volume, int):
            raise ValueError("Volume must be an integer.")
        if volume < 0:
            raise ValueError("Volume must be a positive integer.")
        self.volume = volume

class StockReservoir:
    def __init__(
            self,
            infer_dataset : InferDataset,
            config,
            initial_trade_date=None):

        self.infer_dataset = infer_dataset
        self.volumes_ticker_as_key = dict()

        self.klines_as_dict_date_as_key = infer_dataset.stock_data.klines_as_dict_date_as_key
        self.all_trade_dates_sorted = sorted(list(self.klines_as_dict_date_as_key.keys()))

        self.next_trade_date_cache = dict()
        self.prev_trade_date_cache = dict()
        for i in range(config['dataset']['sequence_length'] - 1, len(self.all_trade_dates_sorted) - 1):
            date_string = self.all_trade_dates_sorted[i]
            next_date_string = self.all_trade_dates_sorted[i + 1]
            
            self.next_trade_date_cache[date_string] = next_date_string
            self.prev_trade_date_cache[next_date_string] = date_string
            
            self.next_trade_date_cache[next_date_string] = None  # The last date has no next date
            self.prev_trade_date_cache[date_string] = None  # The first date has no previous date

        if initial_trade_date is None:
            initial_trade_date = self.all_trade_dates_sorted[config['dataset']['sequence_length'] - 1]

        if initial_trade_date not in self.klines_as_dict_date_as_key:
            raise ValueError(f"Initial trade date {initial_trade_date} is not in the dataset range.")
        self.trade_date = initial_trade_date

        self.human_defect_level = config['stock_exchange_agent']['human_defect_level']
        self.exchange_near_close_time = config['stock_exchange_agent']['exchange_near_close_time']
    

    def __random_buying_price(self, min, max):

        price = min + (max - min) * self.human_defect_level
        live_price = self.__random_price_around_a_price(min, max, price)

        return live_price
    

    def __random_selling_price(self, min, max):

        price = min + (max - min) * (1.0 - self.human_defect_level)
        live_price = self.__random_price_around_a_price(min, max, price)

        return live_price


    def __random_price_around_a_price(self, min_price, max_price, price):
        if min_price >= max_price:
            #raise ValueError("Invalid price range.")
            return price

        # Generate a gaussian random value price as mean
        # sigma is max(max - price, price - min) / 3
        sigma = max(max_price - price, price - min_price) / 3
        random_price = np.random.normal(price, sigma)

        # Ensure the random price is within the specified range
        while random_price < min_price or random_price > max_price:
            random_price = np.random.normal(price, sigma)

        return random_price


    def __close_price_of_prev_trade_date(self, ticker):
        if self.trade_date not in self.prev_trade_date_cache:
            return None

        prev_trade_date = self.prev_trade_date_cache[self.trade_date]
        if prev_trade_date not in self.klines_as_dict_date_as_key:
            return None

        kline_tickers_as_key = self.klines_as_dict_date_as_key[prev_trade_date]
        if ticker not in kline_tickers_as_key:
            return None

        return kline_tickers_as_key[ticker]['closePrice']


    def __calculate_closest_volume_and_amount_according_to_ticker_price(self, amount : Decimal, price : Decimal):

        # Calculate the closest amount according to the price
        # For example, if the price is 15.2 and the amount is 100,000 then 100,000 / 15.2 = 6578.947368421052 then up half to 6600
        # Then the closest amount is 6600 * 15.2 = 100,320
        # volume should be rounded half up to the nearest integer which is times of 100

        volume = amount / price
        # Round the volume to the nearest integer that is a multiple of 100
        volume = round(volume / 100) * 100
        amount = volume * Decimal(price)
        return volume, amount
    

    def estimate_volume_to_buy(self, ticker, amount : Decimal):
        if not isinstance(amount, Decimal):
            raise ValueError("Amount must be a Decimal.")

        tickers_available = self.infer_dataset.get_all_tickers_of_date(self.trade_date)
        if ticker not in tickers_available or ticker not in self.klines_as_dict_date_as_key[self.trade_date]:
            raise ValueError(f"Ticker {ticker} is not available on trade date {self.trade_date}.")

        kline_tickers_as_key = self.klines_as_dict_date_as_key[self.trade_date]

        price = kline_tickers_as_key[ticker]['closePrice']
        price = round(price, 2)
        volume, amount = self.__calculate_closest_volume_and_amount_according_to_ticker_price(amount, Decimal(price))

        return volume, amount


    def buy(self, personal_reservoir, ticker, volume : int, ):

        if not volume > 0:
            return

        tickers_available = self.infer_dataset.get_all_tickers_sample_available_on_date(self.trade_date)
        if ticker not in tickers_available or ticker not in self.klines_as_dict_date_as_key[self.trade_date]:
            raise ValueError(f"Ticker {ticker} is not available or not recommended on trade date {self.trade_date}.")

        kline_tickers_as_key = self.klines_as_dict_date_as_key[self.trade_date]
        live_price_max = kline_tickers_as_key[ticker]['highestPrice']
        live_price_min = kline_tickers_as_key[ticker]['lowestPrice']
        close_price = kline_tickers_as_key[ticker]['closePrice']

        if self.exchange_near_close_time:
            live_price = self.__random_price_around_a_price(live_price_min, live_price_max, close_price)
        else:
            live_price = self.__random_buying_price(live_price_min, live_price_max)

        # Price has only two decimal places
        live_price = round(live_price, 2)

        amount = volume * Decimal(live_price)
        available_amount = personal_reservoir.get_amount()
        if available_amount < amount:
            raise ValueError(f"Not enough money to buy {ticker}. Available amount: {available_amount}, required amount: {amount}.")
        
        if ticker not in self.volumes_ticker_as_key:
            self.volumes_ticker_as_key[ticker] = TickerReservoir(0)

        personal_reservoir.withdraw(amount)
        self.volumes_ticker_as_key[ticker].save(volume)

        print_log(
            f"Bought {ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.trade_date)} for {amount :>10.2f} {volume:>6} * {live_price:.2f}",
            level='INFO')


    def sell(self, personal_reservoir, ticker):

        tickers_available = self.infer_dataset.get_all_tickers_sample_available_on_date(self.trade_date)
        if ticker not in tickers_available or ticker not in self.klines_as_dict_date_as_key[self.trade_date]:
            raise ValueError(f"Ticker {ticker} is not available or not recommended on trade date {self.trade_date}.")

        if ticker not in self.volumes_ticker_as_key:
            raise ValueError(f"Ticker {ticker} is not in the reservoir.")
        
        volume = self.volumes_ticker_as_key[ticker].get_volume()
        volume_sold = self.volumes_ticker_as_key[ticker].withdraw(volume)

        # Check if the volume in reservoir is 0 after selling
        # It it is, remove the ticker from the reservoir
        if self.volumes_ticker_as_key[ticker].get_volume() == 0:
            del self.volumes_ticker_as_key[ticker]

        # The amount sold is based on the live price of the ticker
        kline_tickers_as_key = self.klines_as_dict_date_as_key[self.trade_date]
        live_price_max = kline_tickers_as_key[ticker]['highestPrice']
        live_price_min = kline_tickers_as_key[ticker]['lowestPrice']
        close_price = kline_tickers_as_key[ticker]['closePrice']

        if self.exchange_near_close_time:
            live_price = self.__random_price_around_a_price(live_price_min, live_price_max, close_price)
        else:
            live_price = self.__random_selling_price(live_price_min, live_price_max)

        # Price has only two decimal places
        live_price = round(live_price, 2)

        amount_sold = volume_sold * Decimal(live_price)

        personal_reservoir.save(amount_sold)

        print_log(
            f"Sold {ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.trade_date)} for {amount_sold :>10.2f} {volume_sold :>6} * {live_price :.2f}",
            level='INFO')

    def get_amount(self):
        
        amount = Decimal(0.00)
        
        for ticker, currency_reservoir in self.volumes_ticker_as_key.items():
            tickers_available = self.infer_dataset.get_all_tickers_of_date(self.trade_date)
            if ticker not in tickers_available or ticker not in self.klines_as_dict_date_as_key[self.trade_date]:
                continue
            kline_tickers_as_key = self.klines_as_dict_date_as_key[self.trade_date]
            price = kline_tickers_as_key[ticker]['closePrice']
            amount += currency_reservoir.get_volume() * Decimal(price)
        return amount
    

    def print_account_info(self, extra_info=''):
        if self.trade_date not in self.klines_as_dict_date_as_key:
            print_log(
                f"Trade date {self.trade_date} is not in the dataset range.",
                level='WARNING')
            return

        msg = '\n'
        for ticker, ticker_reservoir in self.volumes_ticker_as_key.items():

            ticker_name = self.infer_dataset.get_name_via_ticker(ticker, self.trade_date)

            tickers_available = self.infer_dataset.get_all_tickers_of_date(self.trade_date)
            if ticker not in tickers_available or ticker not in self.klines_as_dict_date_as_key[self.trade_date]:
                msg += f"{ticker} {ticker_name} is not available on trade date {self.trade_date}.\n"
                continue

            kline_tickers_as_key = self.klines_as_dict_date_as_key[self.trade_date]
            price = kline_tickers_as_key[ticker]['closePrice']
            amount = ticker_reservoir.get_volume() * Decimal(price)
            msg += f"{ticker} {ticker_name}\t{amount:>10.2f} {ticker_reservoir.get_volume():>6} * {price:>6.2f}\n"

        print_log(msg, level='INFO')


    def goto_next_trade_date(self):
        if self.trade_date not in self.next_trade_date_cache:
            self.trade_date = None
            return None

        next_trade_date = self.next_trade_date_cache[self.trade_date]
        if next_trade_date is None:
            self.trade_date = None
            return None

        # Go though each ticker in the reservoir to check if there is XD, XR, DR (XD + XR), R events
        for ticker, ticker_reservoir in self.volumes_ticker_as_key.items():
            if ticker not in self.infer_dataset.get_all_tickers_of_date(next_trade_date):
                print_log(
                    f"Ticker {ticker} is not available on next trade date {next_trade_date}. Skipping.",
                    level='DEBUG')
                continue

            kline_item = self.klines_as_dict_date_as_key[next_trade_date][ticker]
            if 'preClosePrice' in kline_item and 'actPreClosePrice' in kline_item:
                pre_close_price = kline_item['preClosePrice']
                act_pre_close_price = kline_item['actPreClosePrice']

                if pre_close_price != act_pre_close_price:
                    # Price is changed from actual pre close price to pre close price, volume must be amended
                    # Amend the volume to keep the amount of money in the reservoir the same
                    # amount = price * volume, so we need to adjust the volume
                    # new_volume = old_volume * (act_pre_close_price / pre_close_price)
                    # We use int() to round down to the nearest integer
                    # This is because we cannot have fractional shares in the reservoir
                    print_log(
                        f"Amending volume of ticker {ticker} from {ticker_reservoir.get_volume()} to "
                        f"{int(ticker_reservoir.get_volume() * (act_pre_close_price / pre_close_price))} "
                        f"due to XD, XR, DR or R event.",
                        level='WARNING')
                    
                    ticker_reservoir.amend_volume(
                        int(ticker_reservoir.get_volume() * (act_pre_close_price / pre_close_price)))
        
        self.trade_date = next_trade_date
        return next_trade_date

    
    def get_next_trade_date(self):
        if self.trade_date not in self.next_trade_date_cache:
            return None

        next_trade_date = self.next_trade_date_cache[self.trade_date]
        return next_trade_date

    def set_trade_date(self, trade_date):
        if trade_date not in self.next_trade_date_cache:
            raise ValueError(f"Trade date {trade_date} is out of range.")
        self.trade_date = trade_date

    def set_end_date(self, end_date):
        if end_date not in self.klines_as_dict_date_as_key:
            raise ValueError(f"End date {end_date} is out of range.")
        self.next_trade_date_cache[end_date] = None