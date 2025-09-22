import torch

class BatchPreprocessor:
    def __init__(self, normalize_method='minmax', differenced=False, ignore_turnover_data=False):
        self.normalize = True
        self.normalize_method = normalize_method
        self.differenced = differenced
        self.ignore_turnover_data = ignore_turnover_data

        #self.sample_print = False

    def _minmax_normalize_stock_data(self, input_data : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        
        # Normalization of price and turnover volume should be separated
        price_max = input_data[:, :, : 4].amax(dim=(1, 2), keepdim=True)
        price_min = input_data[:, :, : 4].amin(dim=(1, 2), keepdim=True)
        turnover_vol_max = input_data[:, :, 4].amax(dim=1, keepdim=True)
        turnover_vol_min = input_data[:, :, 4].amin(dim=1, keepdim=True)

        # Normalize the input data
        input_data[:, :, : 4] = (input_data[:, :, : 4] - price_min) / ((price_max - price_min) + 1e-20)
        input_data[:, :, 4] = (input_data[:, :, 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

        return input_data
    

    def _minmax_normalize_index_data(self, input_data : torch.Tensor):
        # Copy the input data to avoid modifying the original data
        #input_data = input_data.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_max = input_data[:, :, i * 5 : i * 5 + 4].amax(dim=(1, 2), keepdim=True)
            price_min = input_data[:, :, i * 5 : i * 5 + 4].amin(dim=(1, 2), keepdim=True)
            turnover_vol_max = input_data[:, :, i * 5 + 4].amax(dim=1, keepdim=True)
            turnover_vol_min = input_data[:, :, i * 5 + 4].amin(dim=1, keepdim=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = (input_data[:, :, i * 5 : i * 5 + 4] - price_min) / ((price_max - price_min) + 1e-20)
            input_data[:, :, i * 5 + 4] = (input_data[:, :, i * 5 + 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

        return input_data
    

    def _zeromax_normalize_stock_data(self, input_data : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()

        # Normalization of price and turnover volume should be separated
        price_max = input_data[:, :, : 4].amax(dim=(1, 2), keepdim=True)
        turnover_vol_max = input_data[:, :, 4].amax(dim=1, keepdim=True)

        # Normalize the input data
        input_data[:, :, : 4] = input_data[:, :, : 4] / (price_max + 1e-20)
        input_data[:, :, 4] = input_data[:, :, 4] / (turnover_vol_max + 1e-20)
        
        return input_data
    

    def _zeromax_normalize_index_data(self, input_data):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_max = input_data[:, :, i * 5 : i * 5 + 4].max(dim=(1, 2), keepdim=True)
            turnover_vol_max = input_data[:, :, i * 5 + 4].max(dim=1, keepdim=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = input_data[:, :, i * 5 : i * 5 + 4] / (price_max + 1e-20)
            input_data[:, :, i * 5 + 4] = input_data[:, :, i * 5 + 4] / (turnover_vol_max + 1e-20)

        return input_data
        

    def _normalize_for_differenced_stock_data(self, input_data : torch.Tensor, last_price : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #last_price = last_price.clone().detach()

        # Normalization of price and turnover volume should be separated
        price_diff_std = input_data[:, :, : 4].std(axis=(1, 2), keepdims=True, unbiased=False)
        turnover_vol_std = input_data[:, :, 4].std(axis=1, keepdims=True, unbiased=False)
        turnover_vol_mean = input_data[:, :, 4].mean(axis=1, keepdims=True)

        # Normalize the input data
        input_data[:, :, : 4] = input_data[:, :, : 4] / (price_diff_std + 1e-20)
        input_data[:, :, 4] = (input_data[:, :, 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

        # Normalize the last price
        last_price = last_price / (price_diff_std.view(-1, 1) + 1e-20)

        last_price = last_price / 100
        last_price = last_price.clip(min=0.0, max=100.0)

        return input_data, last_price
    

    def _normalize_for_differenced_index_data(self, input_data : torch.Tensor, last_price : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #last_price = last_price.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_diff_std = input_data[:, :, i * 5 : i * 5 + 4].std(axis=(1, 2), keepdims=True, unbiased=False)
            turnover_vol_std = input_data[:, :, i * 5 + 4].std(axis=1, keepdims=True, unbiased=False)
            turnover_vol_mean = input_data[:, :, i * 5 + 4].mean(axis=1, keepdims=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = input_data[:, :, i * 5 : i * 5 + 4] / (price_diff_std + 1e-20)
            input_data[:, :, i * 5 + 4] = (input_data[:, :, i * 5 + 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

            if last_price is not None:
                # Normalize the last price
                last_price[:, i * 4 : (i + 1) * 4] = last_price[:, i * 4 : (i + 1) * 4] / (price_diff_std.view(-1, 1) + 1e-20)

        last_price = last_price / 100
        last_price = last_price.clip(min=0.0, max=100.0)

        return input_data, last_price

                
    def _normalize_stock_data(self, input_data, normalize_method='minmax'):

        if normalize_method == 'minmax':
            # Normalize the input data and target data
            input_data_n = self._minmax_normalize_stock_data(input_data)

        elif normalize_method == 'zeromax':
            # Normalize the input data and target data
            input_data_n = self._zeromax_normalize_stock_data(input_data)
        else:
            raise ValueError(f"Unsupported normalization method for non-differenced data: {normalize_method}. Supported methods are 'minmax, zeromax'.")

        return input_data_n
    

    def _normalize_index_data(self, input_data, normalize_method='minmax'):

        if normalize_method == 'minmax':
            # Normalize the input data
            input_data_n = self._minmax_normalize_index_data(input_data)

        elif normalize_method == 'zeromax':
            # Normalize the input data
            input_data_n = self._zeromax_normalize_index_data(input_data)
        else:
            raise ValueError(f"Unsupported normalization method for non-differenced data: {normalize_method}. Supported methods are 'minmax, zeromax'.")

        return input_data_n
        

    def _difference_stock_data(self, stock_input_data : torch.Tensor):
        
        last_price = stock_input_data[:, -1, :-1] # last price of the input data

        # Calculate diff of prices part
        # Array length should be shortened by 1
        stock_data_diff = torch.diff(stock_input_data[:, :, :-1], dim=1)
        # Concatenate the diff of prices and turnover volume
        stock_data_diff = torch.cat((stock_data_diff, stock_input_data[:, 1:, -1:]), dim=2)


        return stock_data_diff, last_price
        

    def _difference_index_data(self, index_input_data : torch.Tensor):

        index_input_data_list = []
        last_price_list = []
        for i in range(index_input_data.size(2) // 5):
            last_price = index_input_data[:, -1, i * 5 : i * 5 + 4] # last price of the input data
            # Calculate diff of prices part
            # Array length should be shortened by 1
            index_input_data_diff = torch.diff(index_input_data[:, :, i * 5 : i * 5 + 4], dim=1)
            # Concatenate the diff of prices and turnover volume
            index_input_data_diff = torch.cat((index_input_data_diff, index_input_data[:, 1:, i * 5 + 4 : (i + 1) * 5]), dim=2)

            index_input_data_list.append(index_input_data_diff)
            last_price_list.append(last_price)

        return torch.cat(index_input_data_list, axis=2), torch.cat(last_price_list, axis=1)
    

    def _clear_turnover_data(self, indices_input_array : torch.Tensor, stock_input_array : torch.Tensor):
        
        for i in range(indices_input_array.size(2) // 5):
            # Clear the turnover data in indices input array
            indices_input_array[:, :, i * 5 + 4] = 0.0
        # Clear the turnover data in stock input array
        stock_input_array[:, :, 4] = 0.0

        return indices_input_array, stock_input_array
    

    def process_sample(self, indices_input_array, stock_input_array):

        if self.differenced:

            # Difference the input data

            indices_input_array, indices_last_price = self._difference_index_data(indices_input_array)

            stock_input_array, stock_last_price = self._difference_stock_data(stock_input_array)

            if self.normalize:
                if self.normalize_method != 'standard':
                    raise ValueError(f"Unsupported normalization method for differenced data: {self.normalize_method}. Supported method is 'standard'.")
                # Normalize the input data
                indices_input_array, indices_last_price = \
                    self._normalize_for_differenced_index_data(indices_input_array, indices_last_price)

                stock_input_array, stock_last_price = \
                    self._normalize_for_differenced_stock_data(stock_input_array, stock_last_price)

            if self.ignore_turnover_data:
                indices_input_array, stock_input_array = self._clear_turnover_data(indices_input_array, stock_input_array)
                
            return indices_input_array, stock_input_array, indices_last_price, stock_last_price

        else:
            if self.normalize:
                # Normalize the input and target data
                indices_input_array = self._normalize_index_data(indices_input_array, self.normalize_method)
                stock_input_array = self._normalize_stock_data(stock_input_array, self.normalize_method)

            if self.ignore_turnover_data:
                indices_input_array, stock_input_array = self._clear_turnover_data(indices_input_array, stock_input_array)

            return indices_input_array, stock_input_array


    def reconstruct_differenced_stock_prices(self, input_prices : torch.Tensor, last_price : torch.Tensor):
        # Reconstruct the stock input data and target data from the differenced data
        
        # input_prices: [batch_size, history_time_steps - 1, features]
        # last_price: [batch_size, features]

        # input_prices_reconstructed: [batch_size, history_time_steps, features]

        last_price = last_price * 100  # Convert last price back to original scale

        # Reconstruct the input data
        input_prices_cumsum = torch.cumsum(input_prices, dim=1)
        first_price = last_price - input_prices_cumsum[:, -1, :]
        first_price = first_price.unsqueeze(1)
        input_prices_reconstructed = first_price + input_prices_cumsum
        input_prices_reconstructed = torch.cat((first_price, input_prices_reconstructed), dim=1)

        return input_prices_reconstructed
    

    def calculate_price_change_rate_via_differenced_data(self, last_price : torch.Tensor, future_seq : torch.Tensor):
        # Calculate the price change rate via differenced data
        # last_price: [batch_size, features]
        # future_seq: [batch_size, future_time_steps, features]

        last_price = last_price * 100  # Convert last price back to the same scale as other difference data

        # pick up close price from last_price
        # last_price: [batch_size, 4] (open, close, high, low)
        # last_price_close: [batch_size, 1]
        last_close_price = last_price[:, 1]

        # Calculate the price change rate
        close_price_change = torch.sum(future_seq[:, :, 1], dim=1)

        price_change_rate = close_price_change / (last_close_price + 1e-20)

        return price_change_rate


class TrainBatchPreprocessor(BatchPreprocessor):
    def __init__(self, normalize_method='minmax', differenced=False, ignore_turnover_data=False):
        super().__init__(normalize_method, differenced, ignore_turnover_data)
    
    def _minmax_normalize_stock_data(self, input_data : torch.Tensor, target_data : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #target_data = target_data.clone().detach()
        
        # Normalization of price and turnover volume should be separated
        price_max = input_data[:, :, : 4].amax(dim=(1, 2), keepdim=True)
        price_min = input_data[:, :, : 4].amin(dim=(1, 2), keepdim=True)
        turnover_vol_max = input_data[:, :, 4].amax(dim=1, keepdim=True)
        turnover_vol_min = input_data[:, :, 4].amin(dim=1, keepdim=True)

        # Normalize the input data
        input_data[:, :, : 4] = (input_data[:, :, : 4] - price_min) / ((price_max - price_min) + 1e-20)
        input_data[:, :, 4] = (input_data[:, :, 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

        # Normalize the target data
        target_data[:, :, : 4] = (target_data[:, :, : 4] - price_min) / ((price_max - price_min) + 1e-20)
        target_data[:, :, 4] = (target_data[:, :, 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

        return input_data, target_data
    

    def _minmax_normalize_index_data(self, input_data : torch.Tensor, target_data : torch.Tensor):
        # Copy the input data to avoid modifying the original data
        #input_data = input_data.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_max = input_data[:, :, i * 5 : i * 5 + 4].amax(dim=(1, 2), keepdim=True)
            price_min = input_data[:, :, i * 5 : i * 5 + 4].amin(dim=(1, 2), keepdim=True)
            turnover_vol_max = input_data[:, :, i * 5 + 4].amax(dim=1, keepdim=True)
            turnover_vol_min = input_data[:, :, i * 5 + 4].amin(dim=1, keepdim=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = (input_data[:, :, i * 5 : i * 5 + 4] - price_min) / ((price_max - price_min) + 1e-20)
            input_data[:, :, i * 5 + 4] = (input_data[:, :, i * 5 + 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

            # Normalize the target data
            target_data[:, :, i * 5 : i * 5 + 4] = (target_data[:, :, i * 5 : i * 5 + 4] - price_min) / ((price_max - price_min) + 1e-20)
            target_data[:, :, i * 5 + 4] = (target_data[:, :, i * 5 + 4] - turnover_vol_min) / ((turnover_vol_max - turnover_vol_min) + 1e-20)

        return input_data, target_data
    

    def _zeromax_normalize_stock_data(self, input_data : torch.Tensor, target_data : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #target_data = target_data.clone().detach()

        # Normalization of price and turnover volume should be separated
        price_max = input_data[:, :, : 4].amax(dim=(1, 2), keepdim=True)
        turnover_vol_max = input_data[:, :, 4].amax(dim=1, keepdim=True)

        # Normalize the input data
        input_data[:, :, : 4] = input_data[:, :, : 4] / (price_max + 1e-20)
        input_data[:, :, 4] = input_data[:, :, 4] / (turnover_vol_max + 1e-20)

        # Normalize the target data
        target_data[:, :, : 4] = target_data[:, :, : 4] / (price_max + 1e-20)
        target_data[:, :, 4] = target_data[:, :, 4] / (turnover_vol_max + 1e-20)
        
        return input_data, target_data
    

    def _zeromax_normalize_index_data(self, input_data : torch.Tensor, target_data : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_max = input_data[:, :, i * 5 : i * 5 + 4].max(dim=(1, 2), keepdim=True)
            turnover_vol_max = input_data[:, :, i * 5 + 4].max(dim=1, keepdim=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = input_data[:, :, i * 5 : i * 5 + 4] / (price_max + 1e-20)
            input_data[:, :, i * 5 + 4] = input_data[:, :, i * 5 + 4] / (turnover_vol_max + 1e-20)

            # Normalize the target data
            target_data[:, :, i * 5 : i * 5 + 4] = target_data[:, :, i * 5 : i * 5 + 4] / (price_max + 1e-20)
            target_data[:, :, i * 5 + 4] = target_data[:, :, i * 5 + 4] / (turnover_vol_max + 1e-20)

        return input_data, target_data
        

    def _normalize_for_differenced_stock_data(self, input_data : torch.Tensor, target_data : torch.Tensor, last_price : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #target_data = target_data.clone().detach()
        #last_price = last_price.clone().detach()

        # Normalization of price and turnover volume should be separated
        price_diff_std = input_data[:, :, : 4].std(axis=(1, 2), keepdims=True, unbiased=False)
        turnover_vol_std = input_data[:, :, 4].std(axis=1, keepdims=True, unbiased=False)
        turnover_vol_mean = input_data[:, :, 4].mean(axis=1, keepdims=True)

        # Normalize the input data
        input_data[:, :, : 4] = input_data[:, :, : 4] / (price_diff_std + 1e-20)
        input_data[:, :, 4] = (input_data[:, :, 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

        # Normalize the target data
        target_data[:, :, : 4] = target_data[:, :, : 4] / (price_diff_std + 1e-20)
        target_data[:, :, 4] = (target_data[:, :, 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

        # Normalize the last price
        last_price = last_price / (price_diff_std.view(-1, 1) + 1e-20)

        last_price = last_price / 100
        last_price = last_price.clip(min=0.0, max=100.0)

        return input_data, target_data, last_price


    def _normalize_for_differenced_index_data(self, input_data : torch.Tensor, target_data : torch.Tensor, last_price : torch.Tensor):
        # Copy the input data and target data to avoid modifying the original data
        #input_data = input_data.clone().detach()
        #last_price = last_price.clone().detach()

        for i in range(input_data.size(2) // 5):
            # Normalization of price and turnover volume should be separated
            price_diff_std = input_data[:, :, i * 5 : i * 5 + 4].std(axis=(1, 2), keepdims=True, unbiased=False)
            turnover_vol_std = input_data[:, :, i * 5 + 4].std(axis=1, keepdims=True, unbiased=False)
            turnover_vol_mean = input_data[:, :, i * 5 + 4].mean(axis=1, keepdims=True)

            # Normalize the input data
            input_data[:, :, i * 5 : i * 5 + 4] = input_data[:, :, i * 5 : i * 5 + 4] / (price_diff_std + 1e-20)
            input_data[:, :, i * 5 + 4] = (input_data[:, :, i * 5 + 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

            # Normalize the target data
            target_data[:, :, i * 5 : i * 5 + 4] = target_data[:, :, i * 5 : i * 5 + 4] / (price_diff_std + 1e-20)
            target_data[:, :, i * 5 + 4] = (target_data[:, :, i * 5 + 4] - turnover_vol_mean) / (turnover_vol_std + 1e-20)

            if last_price is not None:
                # Normalize the last price
                last_price[:, i * 4 : (i + 1) * 4] = last_price[:, i * 4 : (i + 1) * 4] / (price_diff_std.view(-1, 1) + 1e-20)

        last_price = last_price / 100
        last_price = last_price.clip(min=0.0, max=100.0)

        return input_data, target_data, last_price

                
    def _normalize_stock_data(self, input_data, target_data, normalize_method='minmax'):

        if normalize_method == 'minmax':
            # Normalize the input data and target data
            input_data_n, target_data_n = self._minmax_normalize_stock_data(input_data, target_data)

        elif normalize_method == 'zeromax':
            # Normalize the input data and target data
            input_data_n, target_data_n = self._zeromax_normalize_stock_data(input_data, target_data)
        else:
            raise ValueError(f"Unsupported normalization method for non-differenced data: {normalize_method}. Supported methods are 'minmax, zeromax'.")

        return input_data_n, target_data_n
    

    def _normalize_index_data(self, input_data, target_data, normalize_method='minmax'):

        if normalize_method == 'minmax':
            # Normalize the input data
            input_data_n, target_data_n = self._minmax_normalize_index_data(input_data, target_data)

        elif normalize_method == 'zeromax':
            # Normalize the input data
            input_data_n, target_data_n = self._zeromax_normalize_index_data(input_data, target_data)
        else:
            raise ValueError(f"Unsupported normalization method for non-differenced data: {normalize_method}. Supported methods are 'minmax, zeromax'.")

        return input_data_n, target_data_n
        

    def _difference_stock_data(self, stock_input_data : torch.Tensor, stock_target_data : torch.Tensor):

        input_t_len = stock_input_data.size(1)
        
        last_price = stock_input_data[:, -1, :-1] # last price of the input data

        stock_data = torch.cat((stock_input_data, stock_target_data), dim=1)
        # Calculate diff of prices part
        # Array length should be shortened by 1
        stock_data_diff = torch.diff(stock_data[:, :, :-1], dim=1)
        # Concatenate the diff of prices and turnover volume
        stock_data_diff = torch.cat((stock_data_diff, stock_data[:, 1:, -1:]), dim=2)

        stock_input_data = stock_data_diff[:, :input_t_len - 1]
        stock_target_data = stock_data_diff[:, input_t_len - 1:]


        return stock_input_data, stock_target_data, last_price


    def _difference_index_data(self, index_input_data : torch.Tensor, index_target_data : torch.Tensor):

        input_t_len = index_input_data.size(1)

        index_data = torch.cat((index_input_data, index_target_data), dim=1)

        index_data_list = []
        last_price_list = []
        for i in range(index_input_data.size(2) // 5):
            last_price = index_input_data[:, -1, i * 5 : i * 5 + 4] # last price of the input data
            # Calculate diff of prices part
            # Array length should be shortened by 1
            index_data_diff = torch.diff(index_data[:, :, i * 5 : i * 5 + 4], dim=1)
            # Concatenate the diff of prices and turnover volume
            index_data_diff = torch.cat((index_data_diff, index_data[:, 1:, i * 5 + 4 : (i + 1) * 5]), dim=2)

            index_data_list.append(index_data_diff)
            last_price_list.append(last_price)

        index_data = torch.cat(index_data_list, axis=2)
        last_price = torch.cat(last_price_list, axis=1)

        index_input_data = index_data[:, :input_t_len - 1]
        index_target_data = index_data[:, input_t_len - 1:]

        return index_input_data, index_target_data, last_price


    def _clear_turnover_data(self, indices_input_array : torch.Tensor, indices_target_array : torch.Tensor, stock_input_array : torch.Tensor, stock_target_array : torch.Tensor):

        for i in range(indices_input_array.size(2) // 5):
            # Clear the turnover data in indices input array
            indices_input_array[:, :, i * 5 + 4] = 0.0
            indices_target_array[:, :, i * 5 + 4] = 0.0
        # Clear the turnover data in stock input array
        stock_input_array[:, :, 4] = 0.0
        stock_target_array[:, :, 4] = 0.0

        return indices_input_array, indices_target_array, stock_input_array, stock_target_array
    

    def process_sample(self, indices_input_array, indices_target_array, stock_input_array, stock_target_array):

        if self.differenced:

            # Difference the input data

            indices_input_array, indices_target_array, indices_last_price = self._difference_index_data(indices_input_array, indices_target_array)

            stock_input_array, stock_target_array, stock_last_price = self._difference_stock_data(stock_input_array, stock_target_array)

            if self.normalize:
                if self.normalize_method != 'standard':
                    raise ValueError(f"Unsupported normalization method for differenced data: {self.normalize_method}. Supported method is 'standard'.")
                # Normalize the input data
                indices_input_array, indices_target_array, indices_last_price = \
                    self._normalize_for_differenced_index_data(indices_input_array, indices_target_array, indices_last_price)

                stock_input_array, stock_target_array, stock_last_price = \
                    self._normalize_for_differenced_stock_data(stock_input_array, stock_target_array, stock_last_price)

            if self.ignore_turnover_data:
                indices_input_array, indices_target_array, stock_input_array, stock_target_array = \
                    self._clear_turnover_data(indices_input_array, indices_target_array, stock_input_array, stock_target_array)

            return indices_input_array, indices_target_array, stock_input_array, stock_target_array, indices_last_price, stock_last_price

        else:
            if self.normalize:
                # Normalize the input and target data
                indices_input_array, indices_target_array = self._normalize_index_data(indices_input_array, indices_target_array, self.normalize_method)
                stock_input_array, stock_target_array = self._normalize_stock_data(stock_input_array, stock_target_array, self.normalize_method)

            if self.ignore_turnover_data:
                indices_input_array, indices_target_array, stock_input_array, stock_target_array = \
                    self._clear_turnover_data(indices_input_array, indices_target_array, stock_input_array, stock_target_array)

            return indices_input_array, indices_target_array, stock_input_array, stock_target_array


    def reconstruct_differenced_stock_prices(self, input_prices : torch.Tensor, target_prices : torch.Tensor, last_price : torch.Tensor):
        # Reconstruct the stock input data and target data from the differenced data
        
        # input_prices: [batch_size, history_time_steps - 1, features]
        # target_prices: [batch_size, future_time_steps, features]
        # last_price: [batch_size, features]

        # input_prices_reconstructed: [batch_size, history_time_steps, features]
        # target_prices_reconstructed: [batch_size, future_time_steps, features]

        last_price = last_price * 100  # Convert last price back to original scale

        # Reconstruct the input data
        input_prices_cumsum = torch.cumsum(input_prices, dim=1)
        first_price = last_price - input_prices_cumsum[:, -1, :]
        first_price = first_price.unsqueeze(1)
        input_prices_reconstructed = first_price + input_prices_cumsum
        input_prices_reconstructed = torch.cat((first_price, input_prices_reconstructed), dim=1)

        # Reconstruct the target data
        last_price = last_price.unsqueeze(1)
        target_prices_cumsum = torch.cumsum(target_prices, dim=1)
        target_prices_reconstructed = last_price + target_prices_cumsum

        return input_prices_reconstructed, target_prices_reconstructed
    
