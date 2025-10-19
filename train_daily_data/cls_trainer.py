import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import datetime

from torch.utils.data import DataLoader

from train_daily_data.cls_dataset import ClsDataset, load_dataset, load_crossed_dataset
from train_daily_data.utils import print_confusion_matrix
from train_daily_data.models.lstm_model import *
from train_daily_data.preprocess_of_batch import TrainBatchPreprocessor
from train_daily_data.model_selection import early_stop
from train_daily_data.global_logger import print_log

class ClsTrainer:
# Assuming you have a dataset class `StockDataset`
    def __init__(self, config_file_name, config, program_start_time):
        """
        Initialize the ClsTrainer with a configuration dictionary.
        :param config: Dictionary containing configuration parameters for training.
        """
        self.config_file_name = config_file_name
        self.config = config
        self.program_start_time = program_start_time
        self.process_id = 0 # default process value, it would be modified by sub-processes

        train_dataset, val_dataset_1, val_dataset_2 = load_crossed_dataset(config)
        self.train_dataset = train_dataset
        self.val_dataset_1 = val_dataset_1
        self.val_dataset_2 = val_dataset_2


    def train(self, model, train_loader : DataLoader, preprocessor : TrainBatchPreprocessor, cls_loss, reg_loss, optimizer, gpu_id=0, differenced=False):
        model.train()
        total_loss_1 = 0
        total_loss_2 = 0
        for i, samples in enumerate(train_loader):

            index_inputs, index_targets, stock_inputs, stock_targets, cls_label, rate_label = samples
            # Move tensors to the specified GPU
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

            # Forward pass
            if differenced:
                cls, rate = model(inputs, last_price)
            else:
                cls, rate = model(inputs)

            loss_1 = cls_loss(cls, cls_label)
            loss_2 = reg_loss(rate, rate_label)

            loss = loss_1 + loss_2 * 10

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_1 += loss_1.detach().item()
            total_loss_2 += loss_2.detach().item()
            if (i + 1) % 100 == 0:
                print_log(f'Step [{i+1}/{len(train_loader)}], Loss: {loss_1.item():.4f}, {loss_2.item():.4f}', level='INFO')
        return total_loss_1 / len(train_loader), total_loss_2 / len(train_loader)


    def eval(self, model, val_loader : DataLoader, preprocessor : TrainBatchPreprocessor, cls_loss, reg_loss, gpu_id=0, differenced=False, n_classes=3):

        if val_loader is None:
            print_log('Validation loader is None, skipping evaluation.', level='INFO')
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
        model.eval()
        total_loss_1 = 0
        total_loss_2 = 0
        total_failures = 0
        total_successes = 0
        
        #cls_confusion_mat = np.zeros((n_classes, n_classes), dtype=int)
        cls_confusion_mat = torch.zeros((n_classes, n_classes), dtype=torch.int32).to('cuda:{}'.format(gpu_id))

        virtual_bonus = 0.0
        virtual_bonus_count = 0

        with torch.no_grad():
            for i, samples in enumerate(val_loader):

                index_inputs, index_targets, stock_inputs, stock_targets, cls_label, rate_label = samples
                # Move tensors to the specified GPU
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

                loss_1 = cls_loss(cls, cls_label)
                loss_2 = reg_loss(rate, rate_label)
                total_loss_1 += loss_1.detach().item()
                total_loss_2 += loss_2.detach().item()

                pred = cls.argmax(dim=1)
                gt = cls_label

                total_successes += torch.sum(pred == gt)
                total_failures += torch.sum(pred != gt)

                # The following code is equivalent to the commented out code
                # The commented out code is more comprehensible but slower
                '''
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
                '''
                # More efficient implementation using tensor operations
                # Update confusion matrix
                for c_1 in range(n_classes):
                    for c_2 in range(n_classes):
                        cls_confusion_mat[c_1, c_2] += torch.sum((gt == c_1) & (pred == c_2)).item()
                # Update virtual bonus
                bonus_signs = (pred > n_classes // 2).float() - (pred < n_classes // 2).float()
                virtual_bonus += torch.sum(bonus_signs * rate_label).item()
                virtual_bonus_count += len(pred)


        print_log('----------------------------------------------------------', level='INFO')
        precision_of_neg, precision_of_pos = print_confusion_matrix(cls_confusion_mat.cpu().numpy())
        virtual_bonus = virtual_bonus / virtual_bonus_count if virtual_bonus_count > 0 else 0.0

        print_log('Precision of negative classes: {:.4f}'.format(precision_of_neg), level='INFO')
        print_log('Precision of positive classes: {:.4f}'.format(precision_of_pos), level='INFO')
        print_log('Virtual bonus: {:.6f}'.format(virtual_bonus), level='INFO')

        loss_1 = total_loss_1 / len(val_loader)
        loss_2 = total_loss_2 / len(val_loader)
        accuracy = total_successes / (total_failures + total_successes)
        return loss_1, loss_2, accuracy, precision_of_neg, precision_of_pos, virtual_bonus


    def save_model(self, model, root_dir, file_name):

        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(root_dir, file_name)) 


    def train_model_main_loop(self):

        print_log('', level='INFO')
        print_log('##########################################################################################', level='INFO')
        print_log('######################### Start training a new model ####################################', level='INFO')
        print_log('##########################################################################################', level='INFO')
        print_log('', level='INFO')

        train_batch_size = self.config['training']['batch_size']
        val_batch_size = self.config['validation']['batch_size']
        num_epochs = self.config['training']['epochs']

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=self.config['training']['dataloader_workers'],
            )
        
        if self.val_dataset_1 is None:
            val_loader_1 = None
        else:
            val_loader_1 = DataLoader(
                dataset=self.val_dataset_1,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=self.config['validation']['dataloader_workers']
                )
        if self.val_dataset_2 is None:
            val_loader_2 = None
        else:
            val_loader_2 = DataLoader(
                dataset=self.val_dataset_2,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=self.config['validation']['dataloader_workers']
                )

        preprocessor = TrainBatchPreprocessor(
            normalize_method=self.config['preprocessing']['normalize_method'],
            differenced=self.config['preprocessing']['differenced'],
            ignore_turnover_data=self.config['preprocessing']['ignore_turnover_data'],)

        if self.config['model']['type'] == 'LSTM':
            model = LSTMClassifierWithPrice(
                input_size=self.config['model']['input_size'],
                last_price_size=self.config['model']['input_size'] * 4 // 5,
                hidden_size=self.config['model']['hidden_size'],
                num_layers=self.config['model']['num_layers'],
                output_size=self.config['model']['output_size'],
                dropout=self.config['model']['dropout']
            ).to('cuda:{}'.format(self.config['device']['gpu_id']))
        elif self.config['model']['type'] == 'CNNLSTMV2':
            model = CNNLSTMClassifierWithPriceV2(
                input_size=self.config['model']['input_size'],
                last_price_size=self.config['model']['input_size'] * 4 // 5,
                hidden_size=self.config['model']['hidden_size'],
                num_layers=self.config['model']['num_layers'],
                output_size=self.config['model']['output_size'],
                dropout=self.config['model']['dropout']
            ).to('cuda:{}'.format(self.config['device']['gpu_id']))
        elif self.config['model']['type'] == 'CNNLSTMV3':
            model = CNNLSTMClassifierWithPriceV3(
                input_size=self.config['model']['input_size'],
                last_price_size=self.config['model']['input_size'] * 4 // 5,
                hidden_size=self.config['model']['hidden_size'],
                num_layers=self.config['model']['num_layers'],
                output_size=self.config['model']['output_size'],
                dropout=self.config['model']['dropout']
            ).to('cuda:{}'.format(self.config['device']['gpu_id']))
        else:
            raise ValueError(f"Unknown model type: {self.config['model']['type']}")

        cls_loss = nn.CrossEntropyLoss()
        reg_loss = nn.L1Loss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config['training']['max_learning_rate'], 
            weight_decay=self.config['training']['weight_decay'])

        if self.config['training']['random_learning_rate']:
            # Use a random learning rate scheduler
            from train_daily_data.learning_rate import RandomLRSchedulerV1
            lr_scheduler = RandomLRSchedulerV1(
                optimizer,
                min_lr=self.config['training']['min_learning_rate'],
                max_lr=self.config['training']['max_learning_rate'])
        else:
            # Learning rate scheduler
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.config['training']['min_learning_rate'] / self.config['training']['max_learning_rate'],
                total_iters=self.config['training']['warmup_epochs'])
            #warmup_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['training']['warmup_gamma'])
            decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['training']['step_size'], gamma=self.config['training']['gamma'])

        #  get current time and format it to yyyy-mm-dd_hh-mm-ss
        #  to create a unique directory for each run
        time_as_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Training loop
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        epoch = 0
        while True:

            print_log('--------------------------------------------------------------------', level='INFO')
            
            val_1_loss_1, val_1_loss_2, val_1_acc, val_1_pre_of_neg, val_1_pre_of_pos, val_1_bonus = self.eval(
                model, val_loader_1, preprocessor, cls_loss, reg_loss,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'],
                n_classes=self.config['dataset']['n_classes'])
            print_log(f'Validation 1 Loss: {val_1_loss_1:.4f}+{val_1_loss_2:.4f}, Validation 1 Accuracy: {val_1_acc:.4f}', level='INFO')

            val_2_loss_1, val_2_loss_2, val_2_acc, val_2_pre_of_neg, val_2_pre_of_pos, val_2_bonus = self.eval(
                model, val_loader_2, preprocessor, cls_loss, reg_loss,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'],
                n_classes=self.config['dataset']['n_classes'])
            print_log(f'Validation 2 Loss: {val_2_loss_1:.4f}+{val_2_loss_2:.4f}, Validation 2 Accuracy: {val_2_acc:.4f}', level='INFO')

            if (epoch + 1) % self.config['output']['save_frequency'] == 0:

                save_dir = self.config['output']['model_checkpoint_dir']
                save_dir = os.path.join(
                    save_dir,
                    self.config_file_name + '_' + self.program_start_time + f'_process_{self.process_id}', time_as_str)

                model_file_name = \
                    'model_{0:03d}_acc_{1:.4f}_{2:.4f}_neg_{3:.4f}_{4:.4f}_pos_{5:.4f}_{6:.4f}_bonus_{7:+.6f}_{8:+.6f}.pth'.format(
                        epoch,
                        val_1_acc, val_2_acc,
                        val_1_pre_of_neg, val_2_pre_of_neg,
                        val_1_pre_of_pos, val_2_pre_of_pos,
                        val_1_bonus, val_2_bonus)
                # Save the model
                self.save_model(model, save_dir, model_file_name)

                if early_stop(save_dir, self.config):
                    break
                
            epoch += 1
            if epoch > num_epochs:
                break

            print_log('============================================================================', level='INFO')
            print_log('============================================================================', level='INFO')
            
            print_log(f'Epoch [{epoch}/{num_epochs}], Learning Rate: {optimizer.param_groups[0]['lr']}', level='INFO')

            train_loss_1, train_loss_2 = self.train(
                model, train_loader, preprocessor, cls_loss, reg_loss, optimizer,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'])
            print_log(f'Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss_1:.4f}+{train_loss_2:.4f}', level='INFO')

            if self.config['training']['random_learning_rate']:
                # Use the random learning rate scheduler
                lr_scheduler.step()
                print_log(f'Random Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}', level='INFO')
            else:
                if optimizer.param_groups[0]['lr'] < self.config['training']['max_learning_rate'] and epoch <= self.config['training']['warmup_epochs']:
                    # Warmup phase
                    warmup_scheduler.step()
                else:
                    # Update the learning rate
                    decay_scheduler.step()
