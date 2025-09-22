import torch
import argparse
import numpy as np
import yaml

from torch.utils.data import DataLoader

from train_daily_data.utils import print_confusion_matrix
from train_daily_data.models.lstm_model import *
from train_daily_data.cls_dataset import load_dataset
from train_daily_data.preprocess_of_batch import TrainBatchPreprocessor
from train_daily_data.global_logger import print_log
from train_daily_data.model_selection import get_models_via_checkpoint_dir, \
    determine_prediction_via_voting, determine_prediction_via_voting_fuse_neg_and_pos


def eval(models, val_loader : DataLoader, preprocessor : TrainBatchPreprocessor, gpu_id=0, differenced=False, n_classes=3):

    for model_time_as_key, model in models.items():
        model.eval()

    total_failures = 0
    total_successes = 0
    
    cls_confusion_mat = np.zeros((n_classes, n_classes), dtype=int)

    virtual_bonus = 0.0; virtual_bonus_buy = 0.0; virtual_bonus_sell = 0.0
    virtual_bonus_count = 0; virtual_bonus_buy_count = 0; virtual_bonus_sell_count = 0

    virtual_bonus_50 = 0.0; virtual_bonus_50_buy = 0.0; virtual_bonus_50_sell = 0.0
    virtual_bonus_50_count = 0; virtual_bonus_50_buy_count = 0; virtual_bonus_50_sell_count = 0

    virtual_bonus_70 = 0.0; virtual_bonus_70_buy = 0.0; virtual_bonus_70_sell = 0.0
    virtual_bonus_70_count = 0; virtual_bonus_70_buy_count = 0; virtual_bonus_70_sell_count = 0

    virtual_bonus_90 = 0.0; virtual_bonus_90_buy = 0.0; virtual_bonus_90_sell = 0.0
    virtual_bonus_90_count = 0; virtual_bonus_90_buy_count = 0; virtual_bonus_90_sell_count = 0

    virtual_bonus_of_extreme_classes = 0.0; virtual_bonus_of_extreme_classes_buy = 0.0; virtual_bonus_of_extreme_classes_sell = 0.0
    virtual_bonus_of_extreme_classes_count = 0; virtual_bonus_of_extreme_classes_buy_count = 0; virtual_bonus_of_extreme_classes_sell_count = 0

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

            cls_list = []
            for time_as_key, model in sorted(models.items(), key=lambda x: x[0]):
                if differenced:
                    cls, rate = model(inputs, last_price)
                else:
                    cls, rate = model(inputs)
                cls_list.append(cls)

            #pred = determine_prediction_via_mean(cls_list)

            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote, rate_of_most_vote = determine_prediction_via_voting(cls_list)

            # Fuse classes into 3 classes: negative, neutral, positive
            # Get final prediction via voting along the first dimension (n_models)
            pred_of_most_vote_3cls, rate_of_most_vote_3cls = determine_prediction_via_voting_fuse_neg_and_pos(cls_list)
            
            gt = cls_label

            total_successes += torch.sum(pred_of_most_vote == gt)
            total_failures += torch.sum(pred_of_most_vote != gt)

            for j in range(len(pred_of_most_vote)):
                pred_cls = pred_of_most_vote[j].item()
                gt_cls = gt[j].item()
                cls_confusion_mat[gt_cls, pred_cls] += 1

                gt_rate = rate_label[j].item()
                if pred_cls == n_classes - 1:
                    virtual_bonus_of_extreme_classes += gt_rate
                    virtual_bonus_of_extreme_classes_count += 1
                    virtual_bonus_of_extreme_classes_buy += gt_rate
                    virtual_bonus_of_extreme_classes_buy_count += 1

                elif pred_cls == 0:
                    virtual_bonus_of_extreme_classes -= gt_rate
                    virtual_bonus_of_extreme_classes_count += 1
                    virtual_bonus_of_extreme_classes_sell -= gt_rate
                    virtual_bonus_of_extreme_classes_sell_count += 1

            for j in range(len(pred_of_most_vote_3cls)):
                pred_cls = pred_of_most_vote_3cls[j].item()
                gt_rate = rate_label[j].item()

                if pred_cls == 2:
                    virtual_bonus += gt_rate
                    virtual_bonus_count += 1
                    virtual_bonus_buy += gt_rate
                    virtual_bonus_buy_count += 1

                    if rate_of_most_vote_3cls[j] > 0.5:
                        virtual_bonus_50 += gt_rate
                        virtual_bonus_50_count += 1
                        virtual_bonus_50_buy += gt_rate
                        virtual_bonus_50_buy_count += 1

                    if rate_of_most_vote_3cls[j] > 0.7:
                        virtual_bonus_70 += gt_rate
                        virtual_bonus_70_count += 1
                        virtual_bonus_70_buy += gt_rate
                        virtual_bonus_70_buy_count += 1

                    if rate_of_most_vote_3cls[j] > 0.9:
                        virtual_bonus_90 += gt_rate
                        virtual_bonus_90_count += 1
                        virtual_bonus_90_buy += gt_rate
                        virtual_bonus_90_buy_count += 1

                    
                elif pred_cls == 0:
                    virtual_bonus -= gt_rate
                    virtual_bonus_count += 1
                    virtual_bonus_sell -= gt_rate
                    virtual_bonus_sell_count += 1

                    if rate_of_most_vote_3cls[j] > 0.5:
                        virtual_bonus_50 -= gt_rate
                        virtual_bonus_50_count += 1
                        virtual_bonus_50_sell -= gt_rate
                        virtual_bonus_50_sell_count += 1

                    if rate_of_most_vote_3cls[j] > 0.7:
                        virtual_bonus_70 -= gt_rate
                        virtual_bonus_70_count += 1
                        virtual_bonus_70_sell -= gt_rate
                        virtual_bonus_70_sell_count += 1

                    if rate_of_most_vote_3cls[j] > 0.9:
                        virtual_bonus_90 -= gt_rate
                        virtual_bonus_90_count += 1
                        virtual_bonus_90_sell -= gt_rate
                        virtual_bonus_90_sell_count += 1

                else:
                    virtual_bonus_count += 1

    
    precision_of_neg, precision_of_pos = print_confusion_matrix(cls_confusion_mat)
    virtual_bonus = virtual_bonus / virtual_bonus_count if virtual_bonus_count > 0 else 0.0
    virtual_bonus_buy = virtual_bonus_buy / virtual_bonus_buy_count if virtual_bonus_buy_count > 0 else 0.0
    virtual_bonus_sell = virtual_bonus_sell / virtual_bonus_sell_count if virtual_bonus_sell_count > 0 else 0.0

    virtual_bonus_50 = virtual_bonus_50 / virtual_bonus_50_count if virtual_bonus_50_count > 0 else 0.0
    virtual_bonus_50_buy = virtual_bonus_50_buy / virtual_bonus_50_buy_count if virtual_bonus_50_buy_count > 0 else 0.0
    virtual_bonus_50_sell = virtual_bonus_50_sell / virtual_bonus_50_sell_count if virtual_bonus_50_sell_count > 0 else 0.0

    virtual_bonus_70 = virtual_bonus_70 / virtual_bonus_70_count if virtual_bonus_70_count > 0 else 0.0
    virtual_bonus_70_buy = virtual_bonus_70_buy / virtual_bonus_70_buy_count if virtual_bonus_70_buy_count > 0 else 0.0
    virtual_bonus_70_sell = virtual_bonus_70_sell / virtual_bonus_70_sell_count if virtual_bonus_70_sell_count > 0 else 0.0

    virtual_bonus_90 = virtual_bonus_90 / virtual_bonus_90_count if virtual_bonus_90_count > 0 else 0.0
    virtual_bonus_90_buy = virtual_bonus_90_buy / virtual_bonus_90_buy_count if virtual_bonus_90_buy_count > 0 else 0.0
    virtual_bonus_90_sell = virtual_bonus_90_sell / virtual_bonus_90_sell_count if virtual_bonus_90_sell_count > 0 else 0.0

    virtual_bonus_of_extreme_classes = virtual_bonus_of_extreme_classes / virtual_bonus_of_extreme_classes_count if virtual_bonus_of_extreme_classes_count > 0 else 0.0
    virtual_bonus_of_extreme_classes_buy = virtual_bonus_of_extreme_classes_buy / virtual_bonus_of_extreme_classes_buy_count if virtual_bonus_of_extreme_classes_buy_count > 0 else 0.0
    virtual_bonus_of_extreme_classes_sell = virtual_bonus_of_extreme_classes_sell / virtual_bonus_of_extreme_classes_sell_count if virtual_bonus_of_extreme_classes_sell_count > 0 else 0.0
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Precision of negative classes: {:.4f}'.format(precision_of_neg), level='INFO')
    print_log('Precision of positive classes: {:.4f}'.format(precision_of_pos), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Virtual bonus: {:.6f}'.format(virtual_bonus), level='INFO')
    print_log('Virtual bonus (buy): {:.6f}'.format(virtual_bonus_buy), level='INFO')
    print_log('Virtual bonus (sell): {:.6f}'.format(virtual_bonus_sell), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Virtual bonus (>50% voting): {:.6f}'.format(virtual_bonus_50), level='INFO')
    print_log('Virtual bonus (>50% voting, buy): {:.6f}'.format(virtual_bonus_50_buy), level='INFO')
    print_log('Virtual bonus (>50% voting, sell): {:.6f}'.format(virtual_bonus_50_sell), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Virtual bonus (>70% voting): {:.6f}'.format(virtual_bonus_70), level='INFO')
    print_log('Virtual bonus (>70% voting, buy): {:.6f}'.format(virtual_bonus_70_buy), level='INFO')
    print_log('Virtual bonus (>70% voting, sell): {:.6f}'.format(virtual_bonus_70_sell), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Virtual bonus (>90% voting): {:.6f}'.format(virtual_bonus_90), level='INFO')
    print_log('Virtual bonus (>90% voting, buy): {:.6f}'.format(virtual_bonus_90_buy), level='INFO')
    print_log('Virtual bonus (>90% voting, sell): {:.6f}'.format(virtual_bonus_90_sell), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    print_log('Virtual bonus of extreme classes: {:.6f}'.format(virtual_bonus_of_extreme_classes), level='INFO')
    print_log('Virtual bonus of extreme classes (buy): {:.6f}'.format(virtual_bonus_of_extreme_classes_buy), level='INFO')
    print_log('Virtual bonus of extreme classes (sell): {:.6f}'.format(virtual_bonus_of_extreme_classes_sell), level='INFO')
    print_log('----------------------------------------------------------', level='INFO')
    accuracy = total_successes / (total_failures + total_successes)
    return accuracy, precision_of_neg, precision_of_pos, virtual_bonus



def val_model_main_loop(checkpoint_dir, config, val_dataset, test_dataset):

    print_log('', level='INFO')
    print_log('##########################################################################################', level='INFO')
    print_log('####################### Start verifying ensamble of best models ##########################', level='INFO')
    print_log('##########################################################################################', level='INFO')
    print_log('', level='INFO')

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
    preprocessor = TrainBatchPreprocessor(
        normalize_method=config['preprocessing']['normalize_method'],
        differenced=config['preprocessing']['differenced'],
    )
    
    models = get_models_via_checkpoint_dir(checkpoint_dir, config=config)

    print_log('--------------------------------------------------------------------', level='INFO')
    
    val_acc, _, _, _ = eval(
        models, val_loader, preprocessor,
        gpu_id=config['device']['gpu_id'],
        differenced=config['preprocessing']['differenced'],
        n_classes=config['dataset']['n_classes'])
    print_log(f'Validation Accuracy: {val_acc:.4f}', level='INFO')
    
    print_log('--------------------------------------------------------------------', level='INFO')

    test_acc, _, _, _ = eval(
        models, test_loader, preprocessor,
        gpu_id=config['device']['gpu_id'],
        differenced=config['preprocessing']['differenced'],
        n_classes=config['dataset']['n_classes'])
    
    print_log(f'Accuracy on testing set: {test_acc:.4f}', level='INFO')

        

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
            '-cd', '--checkpoint_dir',
            type=str,
            help='path of model file',
            default='checkpoints_with_bonus'
        )

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    config, _, val_dataset, test_dataset = load_dataset(config)

    val_model_main_loop(
        args.checkpoint_dir,
        config,
        val_dataset,
        test_dataset)