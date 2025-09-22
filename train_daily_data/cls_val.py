import torch
import argparse
import yaml

from utils import print_confusion_matrix
from torch.utils.data import DataLoader
from models.lstm_model import \
    LSTMClassifierWithPrice, \
        CNNLSTMClassifierWithPriceV1, \
            CNNLSTMClassifierWithPriceV2, \
                CNNLSTMClassifierWithPriceV3

from cls_dataset import load_dataset
from preprocess_of_batch import TrainBatchPreprocessor

import os
import numpy as np

# Assuming you have a dataset class `StockDataset`


def eval(model, val_loader : DataLoader, preprocessor : TrainBatchPreprocessor, gpu_id=0, differenced=False, n_classes=3):
    model.eval()
    total_failures = 0
    total_successes = 0
    
    cls_confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

    virtual_bonus = 0.0
    virtual_bonus_count = 0

    with torch.no_grad():
        for i, samples in enumerate(val_loader):

            index_inputs, index_targets, stock_inputs, stock_targets, cls_label, rate_label = samples
            index_inputs = index_inputs.to('cuda:{}'.format(gpu_id))
            index_targets = index_targets.to('cuda:{}'.format(gpu_id))
            stock_inputs = stock_inputs.to('cuda:{}'.format(gpu_id))
            stock_targets = stock_targets.to('cuda:{}'.format(gpu_id))
            cls_label = cls_label.to('cuda:{}'.format(gpu_id))
            rate_label = rate_label.view(-1, 1).to('cuda:{}'.format(gpu_id))

            if differenced:
                index_inputs, index_targets, stock_inputs, stock_targets, last_index_price, last_stock_price = \
                    preprocessor.process_sample(index_inputs, index_targets, stock_inputs, stock_targets)

                last_price = torch.cat((last_index_price, last_stock_price), dim=1)

            else:
                index_inputs, index_targets, stock_inputs, stock_targets = \
                    preprocessor.process_sample(index_inputs, index_targets, stock_inputs, stock_targets)

            inputs = torch.cat((index_inputs, stock_inputs), dim=2)

            if differenced:
                cls, rate = model(inputs, last_price)
            else:
                cls, rate = model(inputs)

            pred = cls.argmax(dim=1)
            gt = cls_label

            total_successes += torch.sum(pred == gt)
            total_failures += torch.sum(pred != gt)

            for j in range(len(pred)):
                pred_cls = pred[j].item()
                gt_cls = gt[j].item()
                cls_confusion_mat[gt_cls, pred_cls] += 1
                gt_rate = rate_label[j].item()

                if pred_cls > n_classes // 2:
                    virtual_bonus += gt_rate
                    virtual_bonus_count += 1
                elif pred_cls < n_classes // 2:
                    virtual_bonus -= gt_rate
                    virtual_bonus_count += 1
                else:
                    virtual_bonus_count += 1


    
    precision_of_neg, precision_of_pos = print_confusion_matrix(cls_confusion_mat)
    virtual_bonus = virtual_bonus / virtual_bonus_count if virtual_bonus_count > 0 else 0.0
    print('----------------------------------------------------------')

    print('Precision of negative classes: {:.4f}'.format(precision_of_neg))
    print('Precision of positive classes: {:.4f}'.format(precision_of_pos))
    print('Virtual bonus: {:.6f}'.format(virtual_bonus))

    accuracy = total_successes / (total_failures + total_successes)
    return accuracy, precision_of_neg, precision_of_pos, virtual_bonus


def val_model_main_loop(model_path, config, val_dataset, test_dataset):

    print()
    print('##########################################################################################')
    print('######################### Start verifying a new model ####################################')
    print('##########################################################################################')
    print()

    val_batch_size = config['testing']['batch_size']

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config['testing']['dataloader_workers']
        )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config['testing']['dataloader_workers']
        )
    
    if config['model']['type'] == 'LSTM':
        model = LSTMClassifierWithPrice(
            input_size=config['model']['input_size'],
            last_price_size=config['model']['input_size'] * 4 // 5,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            output_size=config['model']['output_size'],
            dropout=config['model']['dropout']
        ).to('cuda:{}'.format(config['device']['gpu_id']))
    elif config['model']['type'] == 'CNNLSTMV2':
        model = CNNLSTMClassifierWithPriceV2(
            input_size=config['model']['input_size'],
            last_price_size=config['model']['input_size'] * 4 // 5,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            output_size=config['model']['output_size'],
            dropout=config['model']['dropout']
        ).to('cuda:{}'.format(config['device']['gpu_id']))
    elif config['model']['type'] == 'CNNLSTMV3':
        model = CNNLSTMClassifierWithPriceV3(
            input_size=config['model']['input_size'],
            last_price_size=config['model']['input_size'] * 4 // 5,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            output_size=config['model']['output_size'],
            dropout=config['model']['dropout']
        ).to('cuda:{}'.format(config['device']['gpu_id']))
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")

    preprocessor = TrainBatchPreprocessor(
        normalize_method=config['preprocessing']['normalize_method'],
        differenced=config['preprocessing']['differenced'],
    )

    # Load the model
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path))
    else:
        print(f'Model file {model_path} does not exist.')
        return
    print('Model loaded successfully.')

    print('--------------------------------------------------------------------')
    
    val_acc, _, _, _ = eval(
        model, val_loader, preprocessor,
        gpu_id=config['device']['gpu_id'],
        differenced=config['preprocessing']['differenced'],
        n_classes=config['dataset']['n_classes'])
    print(f'Validation Accuracy: {val_acc:.4f}')
    
    print('--------------------------------------------------------------------')

    test_acc, _, _, _ = eval(
        model, test_loader, preprocessor,
        gpu_id=config['device']['gpu_id'],
        differenced=config['preprocessing']['differenced'],
        n_classes=config['dataset']['n_classes'])
    
    print(f'Accuracy on testing set: {test_acc:.4f}')

        

if __name__ == '__main__': 


    parser = argparse.ArgumentParser(
            description='A sample script demonstrating argparse usage',
            epilog='Example: python script.py -n John --age 25'
        )

    parser.add_argument(
            '-c', '--config',
            type=str,
            help='path of config file',
            default='cls_config.yaml'
        )
    
    parser.add_argument(
            '-m', '--model',
            type=str,
            help='path of model file',
            default='cls_model.pth'
        )

    args = parser.parse_args()

    config, _, val_dataset, test_dataset = load_dataset(args.config)

    val_model_main_loop(
        args.model,
        config,
        val_dataset,
        test_dataset)