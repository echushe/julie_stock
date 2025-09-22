import torch
import torch.nn as nn

class LSTMAutoregress(nn.Module):
    def __init__(self, input_size, hidden_size, future_seq_len, num_layers=1):
        super(LSTMAutoregress, self).__init__()
        self.input_size = input_size  # feature_size
        self.hidden_size = hidden_size
        self.future_seq_len = future_seq_len
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)

        # Fully connected layer to map LSTM hidden state to output feature_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)  # output_size is same as input_size
        )

    def forward(self, x):
        # Input x shape: (batch, historical_seq_len, feature_size)
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Process historical sequence through LSTM
        # lstm_out shape: (batch, historical_seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Initialize output tensor to store predictions
        outputs = torch.zeros(batch_size, self.future_seq_len, self.output_size).to(device)

        # Initialize input for the first future step (last historical input)
        current_input = x[:, -1, :].unsqueeze(1)  # Shape: (batch, 1, feature_size)

        # Autoregressive generation of future sequence
        for t in range(self.future_seq_len):
            # Pass current_input through LSTM
            # lstm_out_t shape: (batch, 1, hidden_size)
            lstm_out_t, (hn, cn) = self.lstm(current_input, (hn, cn))

            # Map LSTM output to prediction
            # prediction shape: (batch, feature_size)
            prediction = self.fc(lstm_out_t[:, -1, :])

            # Store prediction
            outputs[:, t, :] = prediction

            # Use prediction as next input (autoregressive)
            current_input = prediction.detach().unsqueeze(1)  # Shape: (batch, 1, feature_size)

        return outputs


# This model is similar to LSTMAutoregress but includes the last price as an additional input.
# This model supports differenced data only where the last price is the last known price of the asset.
class LSTMAutoregressWithPrice(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, future_seq_len, num_layers=1):
        super(LSTMAutoregressWithPrice, self).__init__()
        self.input_size = input_size  # feature_size
        self.last_price_size = last_price_size
        self.hidden_size = hidden_size
        self.future_seq_len = future_seq_len
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)

        # Fully connected layer to map LSTM hidden state to output feature_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + last_price_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, last_price):
        # Input x shape: (batch, historical_seq_len, feature_size)
        batch_size = x.size(0)
        device = x.device

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Process historical sequence through LSTM
        # lstm_out shape: (batch, historical_seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # Initialize output tensor to store predictions
        outputs = torch.zeros(batch_size, self.future_seq_len, self.input_size).to(device)

        # Initialize input for the first future step (last historical input)
        current_input = x[:, -1, :].unsqueeze(1)  # Shape: (batch, 1, feature_size)

        # Autoregressive generation of future sequence
        for t in range(self.future_seq_len):
            # Pass current_input through LSTM
            # lstm_out_t shape: (batch, 1, hidden_size)
            lstm_out_t, (hn, cn) = self.lstm(current_input, (hn, cn))

            # Concatenate LSTM output with last price
            # last_price shape: (batch, last_price_size)
            fc_input = torch.cat((lstm_out_t[:, -1, :], last_price), dim=1)

            # Map LSTM output to prediction
            # prediction shape: (batch, feature_size)
            prediction = self.fc(fc_input)

            # Store prediction (traces of gradient are saved for loss function in training)
            outputs[:, t, :] = prediction

            # Use prediction as next input (autoregressive, traces of gradient should be removed)
            current_input = prediction.detach().unsqueeze(1)  # Shape: (batch, 1, feature_size)

            # Update last_price for the next step by adding the prediction (traces of gradient should be removed)
            # if feature_size is divisible by 5, we should remove each 5th feature (e.g., volume)
            # if feature_size is divisible by 4, we should use all features
            if self.input_size % 5 == 0:
                # pickup 0-3, 5-8, 10-13, ...
                for i in range(self.input_size // 5):
                    last_price[:, i * 4: (i + 1) * 4] = last_price[:, i * 4: (i + 1) * 4] + prediction.detach()[:, i * 5: i * 5 + 4] / 100

            elif self.input_size % 4 == 0:
                last_price = last_price + prediction.detach() / 100

            else:
                raise ValueError("input_size must be divisible by 4 or 5 for this model.")

        return outputs
