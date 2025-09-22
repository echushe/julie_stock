
import torch.nn as nn

class SimpleModel (nn.Module):

    def __init__(self, input_t_len=10, output_t_len=2, input_size=5, output_size=5):
        super(SimpleModel, self).__init__()

        self.input_t_len = input_t_len
        self.output_t_len = output_t_len
        self.input_size = input_size
        self.output_size = output_size

        #self.linear_1 = nn.Linear(in_features = input_t_len * 5, out_features = output_t_len * 5)

        self.linear_1 = nn.Linear(in_features = input_t_len * self.input_size, out_features = 100)
        self.act_1 = nn.ReLU()
        self.linear_2 = nn.Linear(in_features= 100, out_features = 100)
        self.act_2 = nn.ReLU()
        self.linear_3 = nn.Linear(in_features= 100, out_features = self.output_t_len * self.output_size)

    def forward(self, x):

        x = x.reshape(-1, self.input_t_len * self.input_size)

        out = self.linear_1(x)
        out = self.act_1(out)
        out = self.linear_2(out)
        out = self.act_2(out)
        out = self.linear_3(out)

        out = out.view(-1, self.output_t_len, self.output_size)

        #clearprint(out)

        return out
    

class SimpleClsModel (nn.Module):

    def __init__(self, input_t_len=10, input_size=5, output_size=2):
        super(SimpleClsModel, self).__init__()

        self.input_t_len = input_t_len
        self.input_size = input_size
        self.output_size = output_size

        #self.linear_1 = nn.Linear(in_features = input_t_len * 5, out_features = output_t_len * 5)

        self.linear_1 = nn.Linear(in_features = input_t_len * self.input_size, out_features = 300)
        self.act_1 = nn.Tanh()
        self.linear_2 = nn.Linear(in_features= 300, out_features = 300)
        self.act_2 = nn.Tanh()
        self.linear_3 = nn.Linear(in_features= 300, out_features = self.output_size)

    def forward(self, x):

        x = x.reshape(-1, self.input_t_len * self.input_size)

        out = self.linear_1(x)
        out = self.act_1(out)
        out = self.linear_2(out)
        out = self.act_2(out)
        out = self.linear_3(out)

        out = out.view(-1, self.output_size)

        #clearprint(out)

        return out
    


class CNNClsModel (nn.Module):

    def __init__(self, input_size=4, output_size=2):
        super(CNNClsModel, self).__init__()

        self.output_size = output_size

        # in: b * 4 * 390

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.act_1 = nn.ReLU()
        self.pool_1 = nn.MaxPool1d(kernel_size=5, stride=5) # b * 32 * 78

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.act_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool1d(kernel_size=3, stride=3) # b * 64 * 26

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.act_3 = nn.ReLU()
        self.pool_3 = nn.MaxPool1d(kernel_size=2, stride=2) # b * 128 * 13

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.act_4 = nn.ReLU()
        self.pool_4 = nn.AdaptiveAvgPool1d(output_size=1) # b * 256 * 1

        self.linear_1 = nn.Linear(in_features=256, out_features=self.output_size)


    def forward(self, x):

        x = x.permute(0, 2, 1)

        out = self.conv1(x)
        out = self.act_1(out)
        out = self.pool_1(out)

        out = self.conv2(out)
        out = self.act_2(out)
        out = self.pool_2(out)

        out = self.conv3(out)
        out = self.act_3(out)
        out = self.pool_3(out)

        out = self.conv4(out)
        out = self.act_4(out)
        out = self.pool_4(out)

        out = out.view(-1, 256)

        out = self.linear_1(out)

        return out