import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, future_steps):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.future_steps = future_steps

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.future_steps * self.output_size)
    
    def forward(self, x):
        # x shape: (batch_size, t1, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, t1, hidden_size)
        # Take the output of the last timestep
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_out)  # (batch_size, output_size)
        out = out.view(-1, self.future_steps, self.output_size)  # (batch_size, future_steps, output_size)
        return out
    

class LSTMClsModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClsModel, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, self.output_size)
    
    def forward(self, x):
        # x shape: (batch_size, t1, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, t1, hidden_size)
        # Take the output of the last timestep
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_out)  # (batch_size, output_size)
        out = out.view(-1, self.output_size)  # (batch_size, output_size)
        return out
    

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=2, dropout=0.2):
        """
        Args:
            input_size (int): Number of features in the input (feature_size)
            hidden_size (int): Number of features in the hidden state of the LSTM
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output classes (default=2 for binary classification)
            dropout (float): Dropout probability for regularization (default=0.2)
        """
        super(LSTMClassifier, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,    # feature_size
            hidden_size=hidden_size,  # Number of hidden units
            num_layers=num_layers,    # Number of LSTM layers
            batch_first=True,         # Input shape: (batch_size, series_length, feature_size)
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers (only if num_layers > 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)  # First dense layer
        self.fc2 = nn.Linear(64, output_size)  # Output layer (2 classes)
        self.fc3 = nn.Linear(64, 1)  # Regression output layer (1 value)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        #self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, series_length, feature_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2)
        """
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        
        # Pass input through LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        # lstm_out shape: (batch_size, series_length, hidden_size)
        last_time_step = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through fully connected layers
        x = self.fc1(last_time_step)  # Shape: (batch_size, 64)
        x = self.relu(x)
        x = self.dropout(x)
        cls = self.fc2(x)  # Shape: (batch_size, output_size)
        rate = self.fc3(x)  # Shape: (batch_size, 1)

        #rate = self.hardtanh(rate)  # Apply hardtanh activation to the regression output

        # Apply softmax activation to the classification output only during inference
        #if not self.training:
        #    cls = torch.softmax(cls, dim=1)  # Apply softmax activation to the classification output
        
        return cls, rate  # cls: (batch_size, output_size), rate: (batch_size, 1)


class LSTMClassifierWithPrice(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, num_layers, output_size=2, dropout=0.2):
        """
        Args:
            input_size (int): Number of features in the input (feature_size)
            hidden_size (int): Number of features in the hidden state of the LSTM
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output classes (default=2 for binary classification)
            dropout (float): Dropout probability for regularization (default=0.2)
        """
        super(LSTMClassifierWithPrice, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,    # feature_size
            hidden_size=hidden_size,  # Number of hidden units
            num_layers=num_layers,    # Number of LSTM layers
            batch_first=True,         # Input shape: (batch_size, series_length, feature_size)
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers (only if num_layers > 1)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size + last_price_size, 64)  # First dense layer
        self.fc2 = nn.Linear(64, output_size)  # Output layer (2 classes)
        self.fc3 = nn.Linear(64, 1)  # Regression output layer (1 value)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        #self.hardtanh = nn.Hardtanh(min_val=-1, max_val=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, last_price):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, series_length, feature_size)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2)
        """
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        device = x.device
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        
        # Pass input through LSTM
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        # lstm_out shape: (batch_size, series_length, hidden_size)
        last_time_step = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Concatenate the last time step output with the last price, the last price is a tensor of shape (batch_size, last_price_size)
        last_time_step = torch.cat((last_time_step, last_price), dim=1)  # Shape: (batch_size, hidden_size + last_price_size)
        
        # Pass through fully connected layers
        x = self.fc1(last_time_step)  # Shape: (batch_size, 64)
        x = self.relu(x)
        x = self.dropout(x)
        cls = self.fc2(x)  # Shape: (batch_size, output_size)
        rate = self.fc3(x)  # Shape: (batch_size, 1)

        #rate = self.hardtanh(rate)  # Apply hardtanh activation to the regression output

        # Apply softmax activation to the classification output only during inference
        #if not self.training:
        #    cls = torch.softmax(cls, dim=1)  # Apply softmax activation to the classification output
        
        return cls, rate  # cls: (batch_size, output_size), rate: (batch_size, 1)


class LSTMClassifierWithPriceEnsemble(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, num_layers, output_size=2, dropout=0.2, n_ensemble=100):
        """
        Args:
            input_size (int): Number of features in the input (feature_size)
            hidden_size (int): Number of features in the hidden state of the LSTM
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output classes (default=2 for binary classification)
            dropout (float): Dropout probability for regularization (default=0.2)
        """
        super(LSTMClassifierWithPriceEnsemble, self).__init__()
        
        self.models = nn.ModuleList()
        for i in range(n_ensemble):
            model = LSTMClassifierWithPrice(
                input_size=input_size,    # feature_size
                last_price_size=last_price_size,
                hidden_size=hidden_size,  # Number of hidden units
                num_layers=num_layers,    # Number of LSTM layers
                output_size=output_size,  # Output layer (2 classes)
                dropout=dropout           # Dropout between layers (only if num_layers > 1)
            )
            self.models.append(model)
        
        
    def forward(self, x, last_price):
        
        cls_out_list = []
        rate_out_list = []
        for model in self.models:
            cls, rate = model(x, last_price)
            cls_out_list.append(cls)
            rate_out_list.append(rate)

        # Concatenate the outputs from all models
        cls_out = torch.stack(cls_out_list, dim=0)  # Shape: (n_ensemble, batch_size, output_size)
        rate_out = torch.stack(rate_out_list, dim=0)  # Shape: (n_ensemble, batch_size, 1)

        return cls_out, rate_out  # cls_out: (n_ensemble, batch_size, output_size), rate_out: (n_ensemble, batch_size, 1)
        

class CNNLSTMClassifierWithPriceV1(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, num_layers, output_size=2, dropout=0.2):
        super(CNNLSTMClassifierWithPriceV1, self).__init__()

        # CNN layer
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(input_size * 2),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_size * 2, out_channels=input_size * 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_size * 4),
            nn.ReLU()
        )
        # LSTM layer
        self.lstm = LSTMClassifierWithPrice(
            input_size=input_size + input_size * 4,
            last_price_size=last_price_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

    def forward(self, x, last_price):
        # x shape: (batch_size, series_length, input_size)
        x1 = x.permute(0, 2, 1)
        # x1 shape: (batch_size, input_size, series_length)
        x1 = self.cnn(x1)
        # x1 shape: (batch_size, input_size, series_length)
        x1 = x1.permute(0, 2, 1)
        # x1 shape: (batch_size, series_length, input_size)
        comp = torch.cat((x, x1), dim=2)

        # comp shape: (batch_size, series_length, input_size + input_size)
        cls, rate = self.lstm(comp, last_price)
        # cls shape: (batch_size, output_size)
        # rate shape: (batch_size, 1)
        return cls, rate
    


class CNNLSTMClassifierWithPriceV2(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(CNNLSTMClassifierWithPriceV2, self).__init__()

        # CNN layer
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=9, padding=4),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=9, padding=4),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
        )
        # LSTM layer
        self.lstm = LSTMClassifierWithPrice(
            input_size=input_size + input_size * 4,
            last_price_size=last_price_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

    def forward(self, x, last_price):
        # x shape: (batch_size, series_length, input_size)
        x_1 = x.permute(0, 2, 1)
        # x shape: (batch_size, input_size, series_length)
        x_cnn1 = self.cnn1(x_1)
        x_cnn2 = self.cnn2(x_1)
        x_cnn3 = self.cnn3(x_1)
        x_cnn4 = self.cnn4(x_1)
        # x_cnn1, x_cnn2, x_cnn3, x_cnn4 shape: (batch_size, input_size, series_length)
        x_cnn = torch.cat((x_cnn1, x_cnn2, x_cnn3, x_cnn4), dim=1)
        # x_cnn shape: (batch_size, input_size * 4, series_length)
        x_cnn = x_cnn.permute(0, 2, 1)
        # x_cnn shape: (batch_size, series_length, input_size * 4)
        comp = torch.cat((x, x_cnn), dim=2)

        # comp shape: (batch_size, series_length, input_size + input_size)
        cls, rate = self.lstm(comp, last_price)
        # rate shape: (batch_size, 1)
        return cls, rate
    
class CNNLSTMClassifierWithPriceV3(nn.Module):
    def __init__(self, input_size, last_price_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(CNNLSTMClassifierWithPriceV3, self).__init__()

        # CNN layer
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=7, padding=3),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=9, padding=4),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=9, padding=4),
            nn.BatchNorm1d(input_size),
            nn.Tanh(),
        )
        # LSTM layer
        self.lstm = LSTMClassifierWithPrice(
            input_size=input_size + input_size * 4,
            last_price_size=last_price_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )

    def forward(self, x, last_price):
        # x shape: (batch_size, series_length, input_size)
        x_1 = x.permute(0, 2, 1)
        # x shape: (batch_size, input_size, series_length)
        x_cnn1 = self.cnn1(x_1)
        x_cnn2 = self.cnn2(x_1)
        x_cnn3 = self.cnn3(x_1)
        x_cnn4 = self.cnn4(x_1)
        # x_cnn1, x_cnn2, x_cnn3, x_cnn4 shape: (batch_size, input_size, series_length)
        x_cnn = torch.cat((x_cnn1, x_cnn2, x_cnn3, x_cnn4), dim=1)
        # x_cnn shape: (batch_size, input_size * 4, series_length)
        x_cnn = x_cnn.permute(0, 2, 1)
        # x_cnn shape: (batch_size, series_length, input_size * 4)
        comp = torch.cat((x, x_cnn), dim=2)

        # comp shape: (batch_size, series_length, input_size + input_size)
        cls, rate = self.lstm(comp, last_price)
        # rate shape: (batch_size, 1)
        return cls, rate