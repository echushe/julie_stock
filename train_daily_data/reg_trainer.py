import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import datetime

from torch.utils.data import DataLoader

from train_daily_data.cls_dataset import ClsDataset, load_dataset, load_crossed_dataset
from train_daily_data.utils import print_confusion_matrix
from train_daily_data.models.lstm_autoregressive import *
from train_daily_data.preprocess_of_batch import TrainBatchPreprocessor
from train_daily_data.model_selection import early_stop
from train_daily_data.global_logger import print_log

class RegTrainer:
# Assuming you have a dataset class `StockDataset`
    def __init__(self, config_file_name, config, program_start_time):
        """
        Initialize the RegTrainer with a configuration dictionary.
        :param config: Dictionary containing configuration parameters for training.
        """
        self.config_file_name = config_file_name
        self.config = config
        self.program_start_time = program_start_time
        self.process_id = 0  # default process value, it would be modified by sub-processes

        train_dataset, val_dataset_1, val_dataset_2 = load_crossed_dataset(config)
        self.train_dataset = train_dataset
        self.val_dataset_1 = val_dataset_1
        self.val_dataset_2 = val_dataset_2


    def train(self, model, train_loader : DataLoader, preprocessor : TrainBatchPreprocessor, reg_loss, optimizer, gpu_id=0, differenced=False):
        if not differenced:
            raise ValueError("Differenced must be True for this model.")
        
        model.train()
        total_loss = 0
        for i, samples in enumerate(train_loader):

            original_index_inputs, original_index_targets, original_stock_inputs, original_stock_targets, cls_label, rate_label = samples
            # Move tensors to the specified GPU
            original_index_inputs = original_index_inputs.to('cuda:{}'.format(gpu_id))
            original_index_targets = original_index_targets.to('cuda:{}'.format(gpu_id))
            original_stock_inputs = original_stock_inputs.to('cuda:{}'.format(gpu_id))
            original_stock_targets = original_stock_targets.to('cuda:{}'.format(gpu_id))
            cls_label = cls_label.to('cuda:{}'.format(gpu_id))
            rate_label = rate_label.to('cuda:{}'.format(gpu_id))

            index_inputs, index_targets, stock_inputs, stock_targets, last_index_price, last_stock_price = \
                preprocessor.process_sample(original_index_inputs, original_index_targets, original_stock_inputs, original_stock_targets)

            last_price = torch.cat((last_index_price, last_stock_price), dim=1)
            inputs = torch.cat((index_inputs, stock_inputs), dim=2)
            targets = torch.cat((index_targets, stock_targets), dim=2)

            # Forward pass
            pred = model(inputs, last_price)

            # Shape of pred: (batch_size, future_seq_len, index_feature_size + stock_feature_size)
            # Shape of targets: (batch_size, future_seq_len, index_feature_size + stock_feature_size)
            loss = reg_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

            if (i + 1) % 100 == 0:
                print_log(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}', level='INFO')
        return total_loss / len(train_loader)


    def eval(self, model, val_loader : DataLoader, preprocessor : TrainBatchPreprocessor, reg_loss, gpu_id=0, differenced=False, n_classes=3):
        
        if val_loader is None:
            print_log('Validation loader is None, skipping evaluation.', level='INFO')
            return 0.0, 0.0

        if not differenced:
            raise ValueError("Differenced must be True for this model.")
        model.eval()
        total_loss = 0

        virtual_bonus = 0.0
        virtual_bonus_count = 0

        with torch.no_grad():
            for i, samples in enumerate(val_loader):

                original_index_inputs, original_index_targets, original_stock_inputs, original_stock_targets, cls_label, rate_label = samples
                # Move tensors to the specified GPU
                original_index_inputs = original_index_inputs.to('cuda:{}'.format(gpu_id))
                original_index_targets = original_index_targets.to('cuda:{}'.format(gpu_id))
                original_stock_inputs = original_stock_inputs.to('cuda:{}'.format(gpu_id))
                original_stock_targets = original_stock_targets.to('cuda:{}'.format(gpu_id))
                cls_label = cls_label.to('cuda:{}'.format(gpu_id))
                rate_label = rate_label.to('cuda:{}'.format(gpu_id))

                index_inputs, index_targets, stock_inputs, stock_targets, last_index_price, last_stock_price = \
                    preprocessor.process_sample(original_index_inputs, original_index_targets, original_stock_inputs, original_stock_targets)

                last_price = torch.cat((last_index_price, last_stock_price), dim=1)
                inputs = torch.cat((index_inputs, stock_inputs), dim=2)
                targets = torch.cat((index_targets, stock_targets), dim=2)

                pred = model(inputs, last_price)
                #print_log('size of pred: {}'.format(pred.size()), level='INFO')

                loss = reg_loss(pred, targets)
                total_loss += loss.detach().item()

                price_change_rate_target = preprocessor.calculate_price_change_rate_via_differenced_data(last_stock_price, stock_targets)
                price_change_rate_pred = preprocessor.calculate_price_change_rate_via_differenced_data(last_stock_price, pred[:,:,-5:])

                #print_log('rate_label: {}'.format(rate_label[:10].tolist()), level='INFO')
                #print_log('price_change_rate_target: {}'.format(price_change_rate_target[:10].tolist()), level='INFO')
                #print_log('rate_label - price_change_rate_target: {}'.format((rate_label - price_change_rate_target)[:10].tolist()), level='INFO')

                # Calculate virtual bonus
                sign_equal = (torch.sign(price_change_rate_target) * torch.sign(price_change_rate_pred)).float()
                #print_log('sign_equal: {}'.format(sign_equal[:10].tolist()), level='INFO')
                virtual_bonus += torch.sum(sign_equal * torch.abs(price_change_rate_target))
                virtual_bonus_count += sign_equal.size(0)

        print_log('----------------------------------------------------------', level='INFO')
        virtual_bonus = virtual_bonus / virtual_bonus_count if virtual_bonus_count > 0 else 0.0
        print_log('Virtual bonus: {:.6f}'.format(virtual_bonus), level='INFO')
        loss = total_loss / len(val_loader)

        return loss, virtual_bonus


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
        val_loader_1 = DataLoader(
            dataset=self.val_dataset_1,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.config['validation']['dataloader_workers']
            )
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
            model = LSTMAutoregressWithPrice(
                input_size=self.config['model']['input_size'],
                last_price_size=self.config['model']['input_size'] * 4 // 5,
                hidden_size=self.config['model']['hidden_size'],
                future_seq_len=self.config['dataset']['target_length'],
                num_layers=self.config['model']['num_layers']
            ).to('cuda:{}'.format(self.config['device']['gpu_id']))
        else:
            raise ValueError(f"Unknown model type: {self.config['model']['type']}")

        #cls_loss = nn.CrossEntropyLoss()
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
        train_loss = 0.0

        epoch = 0
        while True:

            print_log('--------------------------------------------------------------------', level='INFO')

            val_1_loss, val_1_bonus = self.eval(
                model, val_loader_1, preprocessor, reg_loss,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'],
                n_classes=self.config['dataset']['n_classes'])
            print_log(f'Validation 1 Loss: {val_1_loss:.4f}, Validation 1 Bonus: {val_1_bonus:.4f}', level='INFO')

            val_2_loss, val_2_bonus = self.eval(
                model, val_loader_2, preprocessor, reg_loss,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'],
                n_classes=self.config['dataset']['n_classes'])
            print_log(f'Validation 2 Loss: {val_2_loss:.4f}, Validation 2 Bonus: {val_2_bonus:.4f}', level='INFO')

            if (epoch + 1) % self.config['output']['save_frequency'] == 0:

                save_dir = self.config['output']['model_checkpoint_dir']
                save_dir = os.path.join(
                    save_dir,
                    self.config_file_name + '_' + self.program_start_time + f'_process_{self.process_id}', time_as_str)

                model_file_name = \
                    'model_{0:03d}_loss_{1:.4f}_{2:.4f}_bonus_{3:+.6f}_{4:+.6f}.pth'.format(
                        epoch,
                        val_1_loss, val_2_loss,
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

            train_loss = self.train(
                model, train_loader, preprocessor, reg_loss, optimizer,
                gpu_id=self.config['device']['gpu_id'],
                differenced=self.config['preprocessing']['differenced'])
            print_log(f'Epoch [{epoch}/{num_epochs}], Training Loss: {train_loss:.4f}', level='INFO')

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
