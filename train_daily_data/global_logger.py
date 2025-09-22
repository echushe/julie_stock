import logging
import datetime
import os


global_logger = logging.getLogger(__name__)

def configure_logger(log_name, config: dict = None, log_to_file: bool = True):
    """
    Configure the global logger with a console handler and a formatter.
    """

    if config is None:
        logging_level = logging.INFO
    else:
        logging_level_str = config['logging']['logging_level']
        if logging_level_str in {'DEBUG', 'debug'}:
            logging_level = logging.DEBUG
        elif logging_level_str in {'INFO', 'info'}:
            logging_level = logging.INFO
        elif logging_level_str in {'WARNING', 'warning'}:
            logging_level = logging.WARNING
        elif logging_level_str in {'ERROR', 'error'}:
            logging_level = logging.ERROR
    
    # Set the logging level for the global logger
    global_logger.setLevel(logging_level)

    if global_logger.hasHandlers():
        # If the logger already has handlers, remove them
        global_logger.handlers.clear()

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    global_logger.addHandler(ch)

    if log_to_file:

        # Create file handler if a log file is specified in the config
        log_file_dir = os.path.join(config['logging']['log_dir'], log_name)

        # get the dir where the log file is located
        if not os.path.exists(log_file_dir):
            # Create the directory if it does not exist
            os.makedirs(log_file_dir)

        time_as_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = os.path.join(log_file_dir, f'{time_as_str}.log')

        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging_level)
        fh.setFormatter(formatter)
        global_logger.addHandler(fh)


def resume_logger(log_file_path, config: dict = None):

    """
    Resume logger from an existing log file
    """

    if config is None:
        logging_level = logging.INFO
    else:
        logging_level_str = config['logging']['logging_level']
        if logging_level_str in {'DEBUG', 'debug'}:
            logging_level = logging.DEBUG
        elif logging_level_str in {'INFO', 'info'}:
            logging_level = logging.INFO
        elif logging_level_str in {'WARNING', 'warning'}:
            logging_level = logging.WARNING
        elif logging_level_str in {'ERROR', 'error'}:
            logging_level = logging.ERROR
    
    # Set the logging level for the global logger
    global_logger.setLevel(logging_level)

    if global_logger.hasHandlers():
        # If the logger already has handlers, remove them
        global_logger.handlers.clear()

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    global_logger.addHandler(ch)

    # get the dir where the log file is located
    if not os.path.exists(log_file_path):
        raise ValueError(f'Log file {log_file_path} to resume from does not exist.')

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging_level)
    fh.setFormatter(formatter)
    global_logger.addHandler(fh)


def print_log(message: str, level: str='INFO',):
    """
    Print a log message with the specified logging level.
    
    Args:
        message (str): The message to log.
        level (str): The logging level (default is 'INFO').
    """
    if level in {'DEBUG', 'debug'}:
        level = logging.DEBUG
    elif level in {'INFO', 'info'}:
        level = logging.INFO
    elif level in {'WARNING', 'warning'}:
        level = logging.WARNING
    elif level in {'ERROR', 'error'}:
        level = logging.ERROR
    else:
        level = logging.INFO
    
    # Log the message
    global_logger.log(level, message)