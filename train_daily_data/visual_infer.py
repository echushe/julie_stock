import torch
import argparse
from torch.utils.data import DataLoader
from cls_dataset import ClsDataset
from models.simple_model import SimpleModel
from models.lstm_model import LSTMModel

import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt


def visual_infer_for_a_ticker(model, val_loader, ticker):

    model.eval()

    for i, (inputs, targets, means, stds, mins, maxs) in enumerate(val_loader):
        outputs = model(inputs.cuda())

        input_c = inputs[0, :, 1].cpu().numpy()
        output_c = outputs[0, :, 1].detach().cpu().numpy()
        target_c = targets[0, :, 1].cpu().numpy()

        gt = np.concatenate((input_c, target_c))
        pred = np.concatenate((input_c, output_c))

        plt.plot(gt, label = "Closing Prices")
        plt.plot(pred, label = "Closing Prices prediction")
        plt.legend()
        plt.title(ticker)
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='A sample script demonstrating argparse usage',
            epilog='Example: python script.py -n John --age 25'
        )

    parser.add_argument(
            '-t', '--ticker',
            type=str,
            help='a nasdaq ticker name',
            default='AAPL'
        )
    
    parser.add_argument(
            '-c', '--checkpoint',
            type=str,
            help='checkpoint file path',
            default='model.pth'
    )

    args = parser.parse_args()
    
    ticker = args.ticker

    input_t_len = 100
    output_t_len = 10

    # Load dataset
    val_dataset = ClsDataset(
        root_dir='polygon/us_daily_data',
        start_date='2024-03-16',
        end_date='2025-03-15',
        ticker=ticker,
        input_t_len=input_t_len,
        target_t_len=output_t_len)
    

    input_size = 4 # o, c, h, l
    output_size = 4 # o, c, h, l

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # Initialize model, loss function, and optimizer
    #model = SimpleModel(input_t_len=input_t_len, output_t_len=output_t_len, input_size=input_size, output_size=output_size).cuda()
    model = LSTMModel(input_size=input_size, hidden_size=100, output_size=output_size, future_steps=output_t_len)
    model.load_state_dict(torch.load(args.checkpoint))
    model.cuda()

    visual_infer_for_a_ticker(model, val_loader, ticker)
