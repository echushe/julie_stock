from train_daily_data.infer_dataset import InferDataset
from train_daily_data.model_cluster import ModelCluster
from exchange_agent_lb import *
from train_daily_data.global_logger import print_log, resume_logger
from decimal import Decimal, ROUND_HALF_UP, getcontext

import random
import json
import os

class StockExchangeAgent:

    def __init__(self, config, infer_dataset : InferDataset, model_cluster : ModelCluster, real_stock_exchange_mode=False):

        getcontext().prec = 16
        getcontext().rounding = ROUND_HALF_UP

        self.config = config
        self.infer_dataset = infer_dataset
        self.model_cluster = model_cluster
        self.n_classes = self.config['dataset']['n_classes']
        self.total_amount_record = []

        self.n_tickers_to_select_each_time = self.config['stock_exchange_agent']['n_tickers_to_select_each_time']

        self.personal_reservoir = CurrencyReservoir(
            Decimal(self.config['stock_exchange_agent']['personal_reservoir_initial_amount']))

        self.stock_reservoir = StockReservoir(self.infer_dataset, self.config)
        
        self.tickers_to_sell, self.tickers_to_buy_dict = [], dict()
        
        self.real_stock_exchange_mode = real_stock_exchange_mode
        if self.real_stock_exchange_mode:
            
            self.agent_state_dir = 'stock_exchange_agent_state'
            
            if not os.path.exists(self.agent_state_dir):
                os.makedirs(self.agent_state_dir)
                print_log(f"Created directory {self.agent_state_dir} for agent state.", level='INFO')
            
            last_date = self.stock_reservoir.all_trade_dates_sorted[-1]  
            # -1 is 'today' (the last downloadable trading date)

            self.stock_reservoir.set_trade_date(last_date)
            agent_state_path = os.path.join(self.agent_state_dir, f'agent_state_{self.config['my_name']}_{last_date}.json')

            if os.path.exists(agent_state_path):
                print_log(f"Loading agent state from {agent_state_path}", level='INFO')
                self.load_agent_state(agent_state_path)
            else:
                print_log(f"No agent state file found for the last date {last_date}. Starting with an empty state.", level='INFO')
                self.create_agent_state(initial_trade_date=last_date)

        else:
            self.agent_state_dir = None


    def create_agent_state(self, initial_trade_date=None):
        """
        Create a default agent state with the initial trade date.
        This is useful for testing purposes.
        """
        if initial_trade_date is None:
            initial_trade_date = self.stock_reservoir.trade_date

        self.stock_reservoir.set_trade_date(initial_trade_date)
        self.stock_reservoir.volumes_ticker_as_key = dict()

        from distill_tickers_from_text import extract_tickers_from_text
        tickers = extract_tickers_from_text()
        for ticker in tickers:
            if ticker not in self.infer_dataset.get_all_tickers_of_date(initial_trade_date):
                print_log(f"Ticker {ticker} is not available on trade date {initial_trade_date}. Skipping.", level='WARNING')
                continue
            if ticker not in self.stock_reservoir.volumes_ticker_as_key:
                volume, amount = self.stock_reservoir.estimate_volume_to_buy(ticker, 
                    Decimal(self.personal_reservoir.get_amount() / max(len(tickers), self.n_tickers_to_select_each_time))
                    )
                self.stock_reservoir.volumes_ticker_as_key[ticker] = TickerReservoir(volume)

        personal_amount = self.personal_reservoir.get_amount() - self.stock_reservoir.get_amount()
        if personal_amount < 0:
            print_log(f"Warning: Personal reservoir amount {personal_amount} is less than 0. Setting it to 0.", level='WARNING')
            personal_amount = Decimal(0)
        self.personal_reservoir.force_set_amount(personal_amount)


    def resume_from_log(self, log_file_path, last_history_date=None):

        if not os.path.exists(log_file_path):
            print_log(f"Log file {log_file_path} does not exist. Cannot resume agent state.", level='ERROR')
            return

        last_trade_date = None
        lines_after_last_trade_date = []
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Amount of total' in line:
                    self.total_amount_record.append(float(line.split()[-1].strip()))
                if 'Date: ' in line:
                    lines_after_last_trade_date = []
                    last_trade_date = line.split()[-1].strip()
                lines_after_last_trade_date.append(line)

        if last_trade_date is None:
            print_log(f"No trade date found in log file {log_file_path}. Cannot resume agent state.", level='ERROR')
            return

        # Extract personal reservoir amount and stock volumes from lines_after_last_trade_date
        lines_of_ticker_volumes = []
        personal_amount = None
        stock_volumes = dict()
        for i, line in enumerate(lines_after_last_trade_date):
            if 'Amount currency:' in line:
                personal_amount = Decimal(line.split()[-1].strip())
            elif 'Amount of purchase in stocks for each ticker' in line:
                lines_of_ticker_volumes = lines_after_last_trade_date[i+1:]
                break
            else:
                continue

        if personal_amount is None:
            print_log(f"Personal reservoir amount not found in log file {log_file_path}. Cannot resume agent state.", level='ERROR')
            return

        lines_of_ticker_to_sell = []
        for i, line in enumerate(lines_of_ticker_volumes):
            if 'Tickers to sell:' in line:
                lines_of_ticker_to_sell = lines_of_ticker_volumes[i+1:]
                break
        
            parts = line.split()

            if len(parts) >= 6:
                # check parts[-3] is number digits or not
                if not parts[-3].isdigit():
                    continue
                ticker = parts[0].strip()
                volume = int(parts[-3].strip())
                stock_volumes[ticker] = volume
            else:
                continue

        tickers_to_sell = []
        lines_of_ticker_to_buy = []
        for i, line in enumerate(lines_of_ticker_to_sell):

            if 'Tickers to buy:' in line:
                lines_of_ticker_to_buy = lines_of_ticker_to_sell[i+1:]
                break

            parts = line.split()

            if len(parts) >= 2:
                ticker = parts[0].strip()
                tickers_to_sell.append(ticker)     
            else:
                continue

        tickers_to_buy_dict = dict()
        for i, line in enumerate(lines_of_ticker_to_buy):
            parts = line.split()
            if len(parts) >= 4:
                 # check parts[-1] is number digits or not
                if not parts[-1].isdigit():
                    continue
                ticker = parts[0].strip()
                volume = int(parts[-1].strip())
                tickers_to_buy_dict[ticker] = volume
            else:
                continue

        if last_history_date is not None and last_trade_date < last_history_date:
            raise ValueError(f"Last trade date {last_trade_date} in log file {log_file_path} is earlier than the specified last history date {last_history_date}.")

        # Now set the agent state
        self.stock_reservoir.set_trade_date(last_trade_date)
        self.personal_reservoir.force_set_amount(personal_amount)
        self.stock_reservoir.volumes_ticker_as_key = dict()
        for ticker, volume in stock_volumes.items():
            self.stock_reservoir.volumes_ticker_as_key[ticker] = TickerReservoir(volume)
        self.tickers_to_sell = tickers_to_sell
        self.tickers_to_buy_dict = tickers_to_buy_dict

        date = self.stock_reservoir.goto_next_trade_date()
        if date is None:
            print_log("No more trading dates available after resuming. Stopping the agent.", level='INFO')
            return


    def print_account_info(self, extra_info=''):

        overview_msg = '\n'

        overview_msg += f'Date: {self.stock_reservoir.trade_date}\n'
        overview_msg += f'{extra_info} Amount of total:  {self.personal_reservoir.get_amount() + self.stock_reservoir.get_amount() :>12.2f}\n'
        overview_msg += f'{extra_info} Amount currency:  {self.personal_reservoir.get_amount() :>12.2f}\n'
        overview_msg += f'{extra_info} Amount in stocks: {self.stock_reservoir.get_amount() :>12.2f}\n'
        print_log(overview_msg, level='INFO')
        print_log(
            f'{extra_info} Amount of purchase in stocks for each ticker:',
            level='INFO')
        self.stock_reservoir.print_account_info(extra_info)
        print_log('', level='INFO')


    def get_account_total_amount(self):
        """
        Get the total amount of the account, including personal reservoir and stock reservoir.
        """
        return self.personal_reservoir.get_amount() + self.stock_reservoir.get_amount()
    

    def save_agent_state(self, state_dir, tickers_to_sell, tickers_to_buy_dict):

        if state_dir is None:
            #print("No state directory specified. Agent state will not be saved.")
            return

        state_dict = {
            'trade_date': self.stock_reservoir.trade_date,
            'personal_reservoir': round(float(self.personal_reservoir.get_amount()), 2),
            'volumes_ticker_as_key': {
                ticker: [self.infer_dataset.get_name_via_ticker(ticker), reservoir.get_volume()]
                for ticker, reservoir in self.stock_reservoir.volumes_ticker_as_key.items()},
            'tickers_to_sell': {
                ticker: self.infer_dataset.get_name_via_ticker(ticker) for ticker in tickers_to_sell},
            'tickers_to_buy': {
                ticker: [self.infer_dataset.get_name_via_ticker(ticker), volume]
                for ticker, volume in tickers_to_buy_dict.items()},
        }
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
            print_log(f"Created directory {state_dir} for saving agent state.", level='INFO')
        # Save the state dictionary to a json file
        state_file_path = f"{state_dir}/agent_state_{self.config['my_name']}_{self.stock_reservoir.trade_date}.json"
        with open(state_file_path, 'w') as f:
            json.dump(state_dict, f, indent=4, ensure_ascii=False)
        print_log(f"Agent state saved to {state_file_path}", level='INFO')


    def load_agent_state(self, state_file_path):
        """
        Load the agent state from a json file.
        The json file should contain the trade date, personal reservoir amount and volumes of each ticker.
        """
        try:
            with open(state_file_path, 'r') as f:
                state_dict = json.load(f)
                
                self.personal_reservoir.force_set_amount(Decimal(state_dict['personal_reservoir']))
                for ticker, (_, volume) in state_dict['volumes_ticker_as_key'].items():
                    self.stock_reservoir.volumes_ticker_as_key[ticker] = TickerReservoir(volume)

                print_log(f"Agent state loaded from {state_file_path}", level='INFO')

        except FileNotFoundError:
            print_log(f"Agent state file {state_file_path} not found.", level='WARNING')
    

    def ticker_filter_rule_already_bought(self, ticker):
        """
        Filter rule 2: Skip tickers that are not in the reservoir.
        """
        if ticker not in self.stock_reservoir.volumes_ticker_as_key:
            return False
        return True
    

    def ticker_filter_rule_turnover_large_enough(self, ticker):
        """
        Filter rule 3: Skip tickers whose daily turnover is too small.
        """
        if ticker not in self.infer_dataset.get_all_tickers_of_date(self.stock_reservoir.trade_date):
            return False
        
        kline_item = self.infer_dataset.stock_data.klines_as_dict_date_as_key[self.stock_reservoir.trade_date][ticker]
        if kline_item['turnoverVol'] * kline_item['closePrice'] < 1e8:  # 100,000 * 1000 = 1e8
            return False
        return True
    

    def ticker_filter_rule_exclude_phasing_out_tickers(self, ticker):
        
        ticker_name = self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)
        if '*ST' in ticker_name or '退市' in ticker_name or '退' in ticker_name:
            return False
        return True
    

    def ticker_filter_rule_exclude_tickers_under_risks(self, ticker):
        ticker_name = self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)
        if '风险' in ticker_name or 'ST' in ticker_name:
            return False
        return True


    def select_tickers_to_buy(self, promising_tickers, n_tickers_to_select_each_time):
        """
        Select tickers to buy from the promising tickers.
        If the number of promising tickers is larger than n_tickers_to_select_each_time,
        randomly select n_tickers_to_select_each_time tickers.
        Otherwise, return all promising tickers.
        """
        if len(promising_tickers) > n_tickers_to_select_each_time:
            if self.real_stock_exchange_mode:
                msg = '\n'
                n_columns = 4
                for i, ticker in enumerate(promising_tickers):
                    volume, amount = self.stock_reservoir.estimate_volume_to_buy(
                        ticker,
                        Decimal((self.personal_reservoir.get_amount() + self.stock_reservoir.get_amount()) / self.n_tickers_to_select_each_time)
                        )

                    ticker_name = self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)
                    msg += f"{ticker} {ticker_name}\t{amount/10000:>4.0f}万 ({volume:>6})                "
                    if (i + 1) % n_columns == 0:
                        msg += '\n'
                print_log(msg, level='INFO')
                # Waiting for terminal reation
                import sys
                import tty
                import termios
                def get_key():
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    try:
                        tty.setraw(sys.stdin.fileno())
                        ch = sys.stdin.read(1)
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    return ch
                print_log("Press enter to confirm random ticker selection or any other key to generate a new random selection.", level='INFO')
                while True:
                    print_log("Generating a new random selection...", level='INFO')
                    selected_tickers = random.sample(promising_tickers, n_tickers_to_select_each_time)
                    print_log("Selected tickers: ", level='INFO')
                    msg = '\n'
                    for ticker in selected_tickers:
                        msg += f"{ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)}\n"
                    print_log(msg, level='INFO')
                    key = get_key()
                    if key == '\n' or key == '\r':
                        print_log("Random ticker selection confirmed.", level='INFO')
                        break
            else:
                selected_tickers = random.sample(promising_tickers, n_tickers_to_select_each_time)
        else:
            selected_tickers = promising_tickers

        random.shuffle(selected_tickers)
        return selected_tickers

    def __predict(self):

        self.print_account_info()

        tickers_to_sell = []
        tickers_can_giveup = []

        date = self.stock_reservoir.trade_date

        original_samples_ticker_as_key = self.infer_dataset.get_original_samples_by_date(date)

        # Special handling for real stock exchange mode on 2025-08-27 (stock market crash), just for logging purpose
        if self.real_stock_exchange_mode and date == '2025-08-27':
            for ticker, original_sample in original_samples_ticker_as_key.items():
                indices_input_array, stock_input_array = original_sample
                if stock_input_array[-1, 1] > stock_input_array[-2, 1]:
                    rate = (stock_input_array[-1, 1] - stock_input_array[-2, 1]) / stock_input_array[-2, 1]
                    # If the closing price of the last day is higher than that of the day before yesterday
                    # It means the stock price is rising
                    ticker_name = self.infer_dataset.get_name_via_ticker(ticker, date)
                    print_log(f"Ticker {ticker} {ticker_name} price is rising: {rate:.2%}", level='INFO')

        # zip original_samples_ticker_as_key into two lists: tickers and processed_samples
        # original_samples_ticker_as_key is a dict with ticker as key and processed samples as value
        # original_samples is a list of original samples
        # tickers is a list of tickers
        # original_samples_ticker_as_key = {'ticker1': original_sample1, 'ticker2': original_sample2, ...}
        # original_samples = [original_sample1, original_sample2, ...]
        # tickers = ['ticker1', 'ticker2', ...]

        tickers, original_samples = [], []
        for ticker, original_sample in original_samples_ticker_as_key.items():

            if self.ticker_filter_rule_already_bought(ticker):

                if not self.ticker_filter_rule_exclude_phasing_out_tickers(ticker):
                    tickers_to_sell.append(ticker)
                    continue
                if not self.ticker_filter_rule_exclude_tickers_under_risks(ticker):
                    tickers_to_sell.append(ticker)
                    continue

                tickers.append(ticker)
                original_samples.append(original_sample)
                
            else:
            
                #if not self.ticker_filter_rule_turnover_large_enough(ticker):
                #    continue

                if not self.ticker_filter_rule_exclude_phasing_out_tickers(ticker):
                    continue
                
                if not self.ticker_filter_rule_exclude_tickers_under_risks(ticker):
                    continue

                tickers.append(ticker)
                original_samples.append(original_sample)

        if self.config['inference']['voting']:

            # Predict using the model cluster
            pred_of_most_vote, rate_of_most_vote, pred_of_most_vote_3cls, rate_of_most_vote_3cls = \
                self.model_cluster.predict(original_samples, infer_batch_size=self.config['inference']['batch_size'])

            promising_tickers = []
            print_log(f'Number of tickers: {len(tickers)}', level='INFO')
            for idx in range(len(tickers)):

                ticker = tickers[idx]
                #print(f"Ticker: {ticker}", end=' ')

                pred_of_most_vote_item = pred_of_most_vote[idx]
                rate_of_most_vote_item = rate_of_most_vote[idx]
                pred_of_most_vote_3cls_item = pred_of_most_vote_3cls[idx]
                rate_of_most_vote_3cls_item = rate_of_most_vote_3cls[idx]

                print_log(
                    f"{ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)}: "
                    f"pred: {pred_of_most_vote_item:>2} ({rate_of_most_vote_item:.2f}), "
                    f"pred_3cls: {pred_of_most_vote_3cls_item:>2} ({rate_of_most_vote_3cls_item:.2f})",
                    level='DEBUG')
                #print(pred_of_most_vote_3cls_item, end=' ')

                if ticker in self.stock_reservoir.volumes_ticker_as_key:
                    if pred_of_most_vote_3cls_item < 2:
                        if pred_of_most_vote_3cls_item < 1 and rate_of_most_vote_3cls_item > 0.7:
                            tickers_to_sell.append(ticker)
                        else:
                            tickers_can_giveup.append(ticker)

                else:
                    if pred_of_most_vote_3cls_item > 1 and rate_of_most_vote_3cls_item > 0.7:
                        promising_tickers.append(ticker)

        else:
            
            # Predict using the model cluster without voting
            pred, pred_3cls = \
                self.model_cluster.predict(original_samples, infer_batch_size=self.config['inference']['batch_size'])

            promising_tickers = []
            print_log(f'Number of tickers: {len(tickers)}', level='INFO')
            for idx in range(len(tickers)):

                ticker = tickers[idx]
                #print(f"Ticker: {ticker}", end=' ')

                pred_item = pred[idx]
                pred_3cls_item = pred_3cls[idx]

                print_log(
                    f"{ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)}: "
                    f"pred: {pred_item:>2}, pred_3cls: {pred_3cls_item:>2}",
                    level='DEBUG')
                #print(pred_3cls_item, end=' ')

                if ticker in self.stock_reservoir.volumes_ticker_as_key:
                    if pred_3cls_item < 2:
                        if pred_3cls_item < 1:
                            tickers_to_sell.append(ticker)
                        else:
                            tickers_can_giveup.append(ticker)
                else:
                    if pred_3cls_item > 1:
                        promising_tickers.append(ticker)


        print_log(f"Number of promising tickers: {len(promising_tickers)}\n", level='INFO')

        # Randomly select some promising tickers
        tickers_to_buy = self.select_tickers_to_buy(promising_tickers, self.n_tickers_to_select_each_time)

        personal_amount_left = self.personal_reservoir.get_amount()
        n_tickers_can_afford = personal_amount_left // \
            ((self.personal_reservoir.get_amount() + self.stock_reservoir.get_amount()) / self.n_tickers_to_select_each_time)
        n_tickers_should_giveup = len(tickers_to_buy) - int(n_tickers_can_afford)

        # In case it is estimated that there is not enough money to buy selected tickers
        if n_tickers_should_giveup > 0:
            if n_tickers_should_giveup >= len(tickers_can_giveup):
                tickers_to_sell.extend(tickers_can_giveup)
            else:
                tickers_to_sell.extend(random.sample(tickers_can_giveup, n_tickers_should_giveup))

        print_log("Tickers to sell: ", level='INFO')

        tickers_to_sell_msg = '\n'
        for ticker in tickers_to_sell:
            tickers_to_sell_msg += f'{ticker} {self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)}\n'
        print_log(tickers_to_sell_msg, level='INFO')

        print_log("Tickers to buy: ", level='INFO')
        tickers_to_buy_dict = dict()

        tickers_to_buy_msg = '\n'
        for ticker in tickers_to_buy:
            volume, amount = self.stock_reservoir.estimate_volume_to_buy(
                ticker,
                Decimal((self.personal_reservoir.get_amount() + self.stock_reservoir.get_amount()) / self.n_tickers_to_select_each_time)
                )
            if not volume > 0:
                # Too high ticker price may cause the volume to be 0
                # Ignore this ticker
                continue

            ticker_name = self.infer_dataset.get_name_via_ticker(ticker, self.stock_reservoir.trade_date)
            tickers_to_buy_msg += f'{ticker} {ticker_name}\t{amount/10000:>4.0f}万 {volume:>6}\n'
            tickers_to_buy_dict[ticker] = volume
        print_log(tickers_to_buy_msg, level='INFO')

        return tickers_to_sell, tickers_to_buy_dict


    def __buy_and_sell(self, tickers_to_sell, tickers_to_buy_dict):

        buying_first = self.config['stock_exchange_agent']['buying_first']
        selling_first = self.config['stock_exchange_agent']['selling_first']

        if buying_first and not selling_first:
            # Buy first, then sell
            for ticker, volume in tickers_to_buy_dict.items():
                try:
                    self.stock_reservoir.buy(self.personal_reservoir, ticker, volume)
                except ValueError as e:
                    pass #print_log(f"Error buying {ticker}: {e}", level='ERROR')

            for ticker in tickers_to_sell:
                try:
                    self.stock_reservoir.sell(self.personal_reservoir, ticker)
                except ValueError as e:
                    pass #print_log(f"Error selling {ticker}: {e}", level='ERROR')

        elif selling_first and not buying_first:
            # Sell first, then buy
            for ticker in tickers_to_sell:
                try:
                    self.stock_reservoir.sell(self.personal_reservoir, ticker)
                except ValueError as e:
                    pass #print_log(f"Error selling {ticker}: {e}", level='ERROR')

            for ticker, volume in tickers_to_buy_dict.items():
                try:
                    self.stock_reservoir.buy(self.personal_reservoir, ticker, volume)
                except ValueError as e:
                    pass #print_log(f"Error buying {ticker}: {e}", level='ERROR')

        else:
            ticker_buy_sell_pairs = list(tickers_to_buy_dict.items()) + [(ticker, None) for ticker in tickers_to_sell]
            # shuffle the pairs to randomize the order of buying and selling
            random.shuffle(ticker_buy_sell_pairs)
            # Buy and sell in the randomized order
            for ticker, volume in ticker_buy_sell_pairs:
                if volume is not None:
                    try:
                        self.stock_reservoir.buy(self.personal_reservoir, ticker, volume)
                    except ValueError as e:
                        pass #print_log(f"Error buying {ticker}: {e}", level='ERROR')
                else:
                    try:
                        self.stock_reservoir.sell(self.personal_reservoir, ticker)
                    except ValueError as e:
                        pass #print_log(f"Error selling {ticker}: {e}", level='ERROR')


    def run(self):

        date = self.stock_reservoir.trade_date       

        while True:

            date, total_amount_record = self.step()

            if date is None:
                print_log("No more trading dates available. Stopping the agent.", level='INFO')
                break

        return total_amount_record


    def step(self):

        if self.stock_reservoir.trade_date is None:
            raise ValueError("Trade date is not set. Please initialize the trade date before calling step().")

        self.__buy_and_sell(self.tickers_to_sell, self.tickers_to_buy_dict)

        self.tickers_to_sell, self.tickers_to_buy_dict = self.__predict()

        total_amount = self.get_account_total_amount()
        self.total_amount_record.append(total_amount)

        self.save_agent_state(self.agent_state_dir, self.tickers_to_sell, self.tickers_to_buy_dict)

        date = self.stock_reservoir.goto_next_trade_date()

        print_log('----------------------------- Next Trading Date ---------------------------------------------\n', level='INFO')

        return date, self.total_amount_record
    

    def set_model_cluster(self, model_cluster : ModelCluster):

        self.model_cluster = model_cluster

    def find_rightful_log_paths(log_dir):
        log_paths = []
        # find all log files in the log_dir (no recursion)
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                log_paths.append(os.path.join(log_dir, file))

        log_paths.sort()

        lines_path_as_key = dict()
        for log_path in log_paths:
            if not os.path.exists(log_path):
                print(f"Log file {log_path} does not exist.")
                continue
            
            with open(log_path, "r") as f:
                lines = f.readlines()
                lines_path_as_key[log_path] = lines

        rightful_log_paths = []
        num_dates = 0

        for log_path, lines in lines_path_as_key.items():
            total_amount_list = []
            date_list_l = []
            for i in range(len(lines)):
                line = lines[i]
                if 'Date: ' in line:
                    date_str = line.split()[-1].strip()
                    next_line = lines[i + 1]

                    date_list_l.append(date_str)
                    total_amount_list.append(float(next_line.split()[-1].strip()))
            
            if len(date_list_l) > num_dates:
                num_dates = len(date_list_l)

            if len(date_list_l) == 0 or len(date_list_l) < num_dates:
                continue

            rightful_log_paths.append(log_path)

        return rightful_log_paths


