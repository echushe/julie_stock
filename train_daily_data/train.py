import torch
import argparse
import yaml
import random
import os
import numpy as np
import datetime
import json
import multiprocessing

from train_daily_data.cls_trainer import ClsTrainer
from train_daily_data.reg_trainer import RegTrainer
from train_daily_data.global_logger import print_log, configure_logger


def constant_seed(seed):
    print_log("Using constant random seed for reproducibility.", level='INFO')
    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # Set seed for CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def variable_seed():
    print_log("Using variable random seed for each run.", level='INFO')
    # Set a random seed based on os.urandom
    random.seed(int.from_bytes(os.urandom(8), 'big'))

    seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        seed = random.randint(0, 2**32 - 1)
        torch.cuda.manual_seed(seed)
        seed = random.randint(0, 2**32 - 1)
        torch.cuda.manual_seed_all(seed)


def training_process_main(trainer : ClsTrainer | RegTrainer, use_constant_seed, num_models, process_id, gpu_id):

    # Set GPU id for the trainer
    trainer.config['device']['gpu_id'] = gpu_id

    for i in range(num_models):

        if not use_constant_seed:
            variable_seed()

        configure_logger(log_name + f'_process_{process_id}', trainer.config)
        print_log(f'################## Process {process_id} GPU:{gpu_id} #### Start training model {i + 1} ##################', level='INFO')
        
        # Train the model
        trainer.process_id = process_id
        trainer.train_model_main_loop()


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
    
    # Argument to use constant random seed or not
    parser.add_argument(
            '-cs', '--use_constant_seed',
            help='specify to use constant random seed',
            action='store_true'
        )
    
    # Argument to specify number of models to train
    parser.add_argument(
            '-n', '--num_models',
            type=int,
            help='number of models to train',
            default=1
        )

    # Specify GPU id
    # This specification will override the GPU id in the config file
    parser.add_argument(
            '-gl', '--gpu_id_list',
            type=str,
            help='Comma-separated list of GPU ids to use for training',
            default='0'
        )

    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_id_list.split(',')]

    program_start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_name = os.path.basename(args.config).replace('.yaml', '') + '_' + program_start_time
    configure_logger(log_name, config)
    print_log(json.dumps(config, indent=4), level='INFO')

    if config['model']['autoregressive']:
        print_log('Using autoregressive model for training.', level='INFO')
        trainer = RegTrainer(os.path.basename(args.config).replace('.yaml', ''), config, program_start_time)
    else:
        trainer = ClsTrainer(os.path.basename(args.config).replace('.yaml', ''), config, program_start_time)

    if args.use_constant_seed:
        # Set a constant seed for reproducibility
        constant_seed(42)

    num_processes = len(gpu_ids)
    if num_processes > 1:
        # Use multiprocessing
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=training_process_main,
                args=(trainer, args.use_constant_seed, args.num_models, i, gpu_ids[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        # Use single process
        training_process_main(trainer, args.use_constant_seed, args.num_models, 0, gpu_ids[0])
