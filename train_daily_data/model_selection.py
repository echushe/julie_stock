import torch
import argparse
import os
import yaml
import math
import copy

from train_daily_data.models.lstm_model import *
from train_daily_data.models.lstm_autoregressive import *
from train_daily_data.global_logger import print_log, configure_logger

class SortMetric:

    def sum_with_penalty(a, b, penalty_factor=0.4):

        if a < 0 or b < 0:
            # If either a or b is negative, return 0
            return 0.0

        reward = a + b
        penalty = abs(a - b)

        return reward - penalty_factor * penalty


    def sum(a, b):

        if a < 0 or b < 0:
            # If either a or b is negative, return 0
            return 0.0

        reward = a + b

        return reward


    def sum_with_balance_1(a, b):

        if a < 0 or b < 0:
            # If either a or b is negative, return 0
            return 0.0

        # This is a more complex metric that considers the ratio of a and b
        a = max(a, 1e-20)  # Avoid division by zero
        b = max(b, 1e-20)  # Avoid division by zero

        return (a + b) / (a / b + b / a)


    def multiply(a, b):

        if a < 0 or b < 0:
            # If either a or b is negative, return 0
            return 0.0

        return a * b


    def harmonic_mean(a, b):

        if a < 0 or b < 0:
            # If either a or b is negative, return 0
            return 0.0
        
        a = max(a, 1e-20)  # Avoid division by zero
        b = max(b, 1e-20)  # Avoid division by zero

        return 2 * a * b / (a + b)  # Harmonic mean


def sort_metric(a, b, metric_name='sum'):
    if metric_name == 'sum_with_penalty':
        return SortMetric.sum_with_penalty(a, b)
    elif metric_name == 'sum':
        return SortMetric.sum(a, b)
    elif metric_name == 'sum_with_balance_1':
        return SortMetric.sum_with_balance_1(a, b)
    elif metric_name == 'multiply':
        return SortMetric.multiply(a, b)
    elif metric_name == 'harmonic_mean':
        return SortMetric.harmonic_mean(a, b)
    else:
        raise ValueError(f'Unknown sort metric: {metric_name}. Available metrics: sum_with_penalty, sum, sum_with_balance_1, multiply, harmonic_mean.')


def early_stop(checkpoint_dir, config):

    patience_epochs = config['training']['patience_epochs']
    performance_metric = config['training']['performance_metric']

    # get all model files in the checkpoint directory
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if len(model_files) == 0:
        print_log(f'No model files found in {checkpoint_dir}', level='WARNING')
        return False
    
    # file name example:
    # model_003_acc_0.3447_0.1852_neg_0.1157_0.5251_pos_0.6185_0.5153_bonus_+0.023972_+0.006722.pth
    
    # pickup epoch numbers and bonus values from model file names
    epoch_performance_pairs = []
    for model_file in model_files:
        model_file_without_ext = model_file.strip('.pth')
        model_file_split = model_file_without_ext.split('_')
        epoch = int(model_file_split[1])
        if model_file_split[-3] == 'bonus':
            bonus_1 = float(model_file_split[-2])
            bonus_2 = float(model_file_split[-1])
            performance = sort_metric(bonus_1, bonus_2, metric_name=performance_metric)
        elif model_file_split[-2] == 'bonus':
            performance= float(model_file_split[-1])

        epoch_performance_pairs.append((epoch, performance))

    # get epoch number of the highest performance
    epoch_performance_pairs.sort(key=lambda x: x[0])  # sort by epoch
    best_epoch = -1
    best_performance = -float('inf')
    for epoch, performance in epoch_performance_pairs:
        if performance > best_performance:
            best_performance = performance
            best_epoch = epoch

    # subtract the best epoch from the last epoch
    last_epoch = epoch_performance_pairs[-1][0]
    epochs_since_best = last_epoch - best_epoch

    if epochs_since_best >= patience_epochs:
        print_log(f'Early stopping triggered. Best epoch: {best_epoch}, Last epoch: {last_epoch}, Epochs since best: {epochs_since_best}', level='INFO')
        return True

    return False


def get_model_paths_via_checkpoint_dir(checkpoint_dir, config):

    # model file format: 
    # model_<0>_acc_<1>_<2>_neg_<3>_<4>_pos_<5>_<6>_bonus_<7>_<8>.pth

    # Find all model files in the checkpoint directory recursively
    models_dir_as_key = dict()

    for root, dirs, files in os.walk(checkpoint_dir):

        for file in files:
            if file.endswith('.pth'):
                file_without_ext = file.strip('.pth')
                file_name_split = file_without_ext.split('_')

                if root in models_dir_as_key:
                    models_dir_as_key[root].append((file, file_name_split))
                else:
                    models_dir_as_key[root] = [(file, file_name_split),]


    best_models_dir_as_key = dict()

    # Go though each time stamp directory
    # and select the best model based on validation bonus
    for root in sorted(models_dir_as_key.keys()):
        model_info_list = models_dir_as_key[root]
        if len(model_info_list) < config['model_pool']['min_checkpoint_idx'] + 1:
            # If there are less than 3 checkpoints, skip this directory
            print_log(f'Not enough checkpoints in {root}, skipping.', level='WARNING')
            continue
        # Select the model with the highest validation bonus (do not consider test bonus)
        # Sort the models based on the validation bonus (second last element in the split file name)

        model_info_list.sort(key=lambda x: float(x[1][1]), reverse=False)
        model_info_list = model_info_list[config['model_pool']['min_checkpoint_idx']:]  # Exclude the 0th, 1th checkpoint

        if model_info_list[0][1][-2] == 'bonus':
            # To get compatible with old model file name format
            model_info_list.sort(key=lambda x: float(x[1][-1]), reverse=True)
        else:
            model_info_list.sort(key=lambda x: sort_metric(float(x[1][-2]), float(x[1][-1]), config['model_pool']['sort_metric']), reverse=True)

        print_log(f'parent dir: {root}', level='INFO')
        for file, file_name_split in model_info_list:
            print_log(f'File: {file}', level='INFO')
        print_log('------------------------------------------------------', level='INFO')

        # select the model of best validation accuracy
        #model_info_list.sort(key=lambda x: float(x[1][3]), reverse=True)

        best = model_info_list[0]

        if best[1][-2] == 'bonus':
            if float(best[1][-1]) < config['model_pool']['min_validation_bonus_of_checkpoints']:
                continue
        else:
            # If the best model has a validation bonus less than min_validation_bonus_of_checkpoints, skip it
            if sort_metric(float(best[1][-2]), float(best[1][-1]), config['model_pool']['sort_metric']) < config['model_pool']['min_validation_bonus_of_checkpoints']:
                #print_log(f'No good model found in {root}, skipping.', level='WARNING')
                continue

        best_models_dir_as_key[root] = best
        #print_log(f'Best model for time stamp {time_stamp} is {model_path} with validation bonus {best[1][-1]}', level='INFO')

    print_log(f'Found {len(models_dir_as_key)} directories with model files for {checkpoint_dir}', level='INFO')

    # Select top models based on validation bonus
    pairs = list(best_models_dir_as_key.items())
    # Sort the pairs based on validation bonus metric
    if len(pairs) > 0 and pairs[0][1][1][-2] == 'bonus':
        # If the model file name format is old, use the last element as validation bonus
        pairs.sort(key=lambda x: float(x[1][1][-1]), reverse=True)
    else:
        pairs.sort(key=lambda x: sort_metric(float(x[1][1][-2]), float(x[1][1][-1]), config['model_pool']['sort_metric']), reverse=True)

    # pickup to n_models_in_each_checkpoint_dir
    pairs = pairs[: config['model_pool']['n_models_in_each_checkpoint_dir']]

    best_model_paths = []
    for dir, (file, file_name_split) in pairs:
        # Get the time stamp from the directory name
        #time_stamp = os.path.basename(dir)
        model_path = os.path.join(dir, file)

        # Check if the model path exists
        if os.path.exists(model_path):
            best_model_paths.append(model_path)
            #print_log(f'Found model {file} in {dir} with validation bonus {file_name_split[-1]}', level='INFO')
        else:
            print_log(f'Model file {model_path} does not exist.', level='WARNING')

    return best_model_paths



def resolve_gpu_plan(n_models, config):

    if n_models == 0:
        raise ValueError('Number of models must be greater than 0.')

    gpu_plan = dict()

    # gpu id is phasing out, use gpu_plan instead
    if 'gpu_id' in config['device']:
        # If gpu_id is specified, use it
        gpu_id = config['device']['gpu_id']
        gpu_plan[gpu_id] = 1
    else:
        # If gpu_id is not specified, use gpu_plan
        gpu_plan = copy.deepcopy(config['device']['gpu_plan'])
        print_log(f'Using gpu_plan: {gpu_plan}', level='INFO')

    if len(gpu_plan) == 0:
        # If gpu_plan is empty, use all available GPUs
        for i in range(torch.cuda.device_count()):
            gpu_plan[i] = 1

    # Sum the weights in gpu_plan
    total_weight = sum(gpu_plan.values())
    # Transform weights into numbers of models per GPU
    for gpu_id in gpu_plan.keys():
        # pickup the number of models for each GPU based on its weight
        float_value = gpu_plan[gpu_id] * n_models / total_weight
        # use ceiling to ensure all models are assigned
        gpu_plan[gpu_id] = math.ceil(float_value)

    return list(gpu_plan.items())


def get_all_model_paths_via_checkpoint_dirs(checkpoint_dirs, config):

    all_model_paths = []
    for checkpoint_dir in checkpoint_dirs:
        all_model_paths.extend(get_model_paths_via_checkpoint_dir(checkpoint_dir, config=config))

    return all_model_paths


def get_models_via_checkpoint_dirs(checkpoint_dirs, config, cpu_mode=False):

    best_model_paths = []
    for checkpoint_dir in checkpoint_dirs:
        best_model_paths.extend(get_model_paths_via_checkpoint_dir(checkpoint_dir, config=config))

    models_path_as_key = get_models_via_paths(best_model_paths, config, cpu_mode)

    return models_path_as_key


def get_models_via_single_model_path(path, config, cpu_mode=False):

    return get_models_via_paths([path], config, cpu_mode)


def get_models_via_paths(best_model_paths, config, cpu_mode=False):

    gpu_plan_queue = resolve_gpu_plan(len(best_model_paths), config)
    gpu_plan_queue_head = gpu_plan_queue.pop(0)

    models_path_as_key = dict()

    # Go though each time stamp directory
    # and select the best model based on validation bonus
    for model_path in best_model_paths:

        while gpu_plan_queue_head[1] == 0:
            # If the head GPU has no more models to load, pop the next one
            if len(gpu_plan_queue) == 0:
                raise ValueError('No more GPUs available in gpu_plan.')
            gpu_plan_queue_head = gpu_plan_queue.pop(0)

        # Decrease the number of models for this GPU
        gpu_id = gpu_plan_queue_head[0]
        gpu_plan_queue_head = (gpu_plan_queue_head[0], gpu_plan_queue_head[1] - 1)

        # Load the model
        if os.path.exists(model_path):
            print_log(f'Loading model from {model_path}', level='info')

            if config['model']['autoregressive']:
                if config['model']['type'] == 'LSTM':
                    model = LSTMAutoregressWithPrice(
                        input_size=config['model']['input_size'],
                        last_price_size=config['model']['input_size'] * 4 // 5,
                        hidden_size=config['model']['hidden_size'],
                        future_seq_len=config['dataset']['target_length'],
                        num_layers=config['model']['num_layers']
                    )
                else:
                    raise ValueError(f"Unknown autoregressive model type: {config['model']['type']}")
                
            else:
                if config['model']['type'] == 'LSTM':
                    model = LSTMClassifierWithPrice(
                        input_size=config['model']['input_size'],
                        last_price_size=config['model']['input_size'] * 4 // 5,
                        hidden_size=config['model']['hidden_size'],
                        num_layers=config['model']['num_layers'],
                        output_size=config['model']['output_size'],
                        dropout=config['model']['dropout']
                    )
                elif config['model']['type'] == 'CNNLSTMV2':
                    model = CNNLSTMClassifierWithPriceV2(
                        input_size=config['model']['input_size'],
                        last_price_size=config['model']['input_size'] * 4 // 5,
                        hidden_size=config['model']['hidden_size'],
                        num_layers=config['model']['num_layers'],
                        output_size=config['model']['output_size'],
                        dropout=config['model']['dropout']
                    )
                elif config['model']['type'] == 'CNNLSTMV3':
                    model = CNNLSTMClassifierWithPriceV3(
                        input_size=config['model']['input_size'],
                        last_price_size=config['model']['input_size'] * 4 // 5,
                        hidden_size=config['model']['hidden_size'],
                        num_layers=config['model']['num_layers'],
                        output_size=config['model']['output_size'],
                        dropout=config['model']['dropout']
                    )
                else:
                    raise ValueError(f"Unknown model type: {config['model']['type']}")
            if not cpu_mode:
                model = model.to('cuda:{}'.format(gpu_id))
                model.load_state_dict(torch.load(model_path, map_location='cuda:{}'.format(gpu_id)))
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            print_log(f'Model file {model_path} does not exist.', level='WARNING')
            continue

        models_path_as_key[model_path] = model

    return models_path_as_key


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
    
    parser.add_argument(
            '-n', '--n_models_in_each_checkpoint_dir',
            type=int,
            help='number of models in each checkpoint directory',
            default=-1
    )
    
    parser.add_argument(
            '-m', '--min_bonus',
            type=float,
            help='minimum validation bonus of checkpoints',
            default=-1.0
        )

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        print(config)

    # overwrite the number of models in each checkpoint directory
    if args.n_models_in_each_checkpoint_dir > 0:
        config['model_pool']['n_models_in_each_checkpoint_dir'] = args.n_models_in_each_checkpoint_dir
        print_log(f'Overwriting number of models in each checkpoint directory to {args.n_models_in_each_checkpoint_dir}', level='INFO')

    # overwrite the minimum validation bonus of checkpoints
    if args.min_bonus >= 0:
        config['model_pool']['min_validation_bonus_of_checkpoints'] = args.min_bonus
        print_log(f'Overwriting minimum validation bonus of checkpoints to {args.min_bonus}', level='INFO')

    # Configure logger
    configure_logger(
        log_name='model_selection',
        config = {
            'logging': {
                'logging_level': 'INFO',
                'log_dir': '/dir/to/my/logs/',
            }
        },
        log_to_file=False,
    )

    model_paths = get_model_paths_via_checkpoint_dir(
        args.checkpoint_dir, config=config)

    for model_path in model_paths:
        path_items = model_path.split('/')
        time_stamp = path_items[-2]
        model_file_name = path_items[-1]
        print_log(f'Time stamp: {time_stamp}, Model: {model_file_name}', level='INFO')

    print_log(f'Found {len(model_paths)} models with validation bonus >= {config["model_pool"]["min_validation_bonus_of_checkpoints"]}', level='INFO')
