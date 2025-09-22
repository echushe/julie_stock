from datetime import datetime
import pytz
from datetime import timedelta
import numpy as np
from train_daily_data.global_logger import print_log

def milliseconds_to_datetime(milliseconds, timezone):
    # Convert milliseconds to seconds by dividing by 1000
    seconds = milliseconds / 1000
    # Create datetime object from timestamp
    dt = datetime.fromtimestamp(seconds, tz=timezone)
    # Format datetime to string
    formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date


def milliseconds_to_date(milliseconds, timezone):
    # Convert milliseconds to seconds by dividing by 1000
    seconds = milliseconds / 1000
    # Create datetime object from timestamp
    dt = datetime.fromtimestamp(seconds, tz=timezone)
    # Format datetime to string
    formatted_date = dt.strftime('%Y-%m-%d')
    return formatted_date


def date_to_milliseconds(date_string, timezone : pytz.timezone):

    # Parse the date string to datetime object
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    # Set the timezone to the datetime object
    date_object_timezone = timezone.localize(date_object)
    date_object_utc = date_object_timezone.astimezone(pytz.utc)
    
    # Convert to milliseconds since epoch
    # First get seconds since epoch
    seconds = date_object_utc.timestamp()
    # Convert to milliseconds by multiplying by 1000
    milliseconds = int(seconds * 1000)
    
    return milliseconds


next_date_cache = dict()
def next_date_of_date(date_string):

    if date_string in next_date_cache:
        return next_date_cache[date_string]

    # Parse the date string to datetime object
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    
    # Add one day to the date
    next_date = date_object + timedelta(days=1)

    # Convert next date to string of format YYYY-MM-DD
    next_date_string = next_date.strftime("%Y-%m-%d")

    # Cache the result
    next_date_cache[date_string] = next_date_string
    
    return next_date_string


def print_confusion_matrix(cls_confusion_mat : np.ndarray):
    """
    Print the confusion matrix for the given true and predicted labels.
    """
    n_classes = cls_confusion_mat.shape[0]

    matrix_output = ''
    
    # Print the header
    matrix_output += "Confusion Matrix:\n"

    number_len_list = []
    number_list = []
    largest_number_digit_len = len(str(cls_confusion_mat.max()))

    matrix_output += "Predicted\t"
    for label in range(n_classes):
        matrix_output +=f'{label:>{largest_number_digit_len}}\t'
    matrix_output += '\n'
    
    # Print the rows of the confusion matrix
    for true_label in range(n_classes):
        matrix_output += f'Expected {true_label}\t'
        for predicted_label in range(n_classes):
            number = cls_confusion_mat[true_label, predicted_label] 
            matrix_output += f'{number:>{largest_number_digit_len}}\t'
        matrix_output += '\n'

    neg_neg = cls_confusion_mat[:n_classes//2, :n_classes//2].sum()
    neg_pos = cls_confusion_mat[:n_classes//2, n_classes//2 + 1:].sum()
    pos_neg = cls_confusion_mat[n_classes//2 + 1:, :n_classes//2].sum()
    pos_pos = cls_confusion_mat[n_classes//2 + 1:, n_classes//2 + 1:].sum()

    neg_of_each_prediction = cls_confusion_mat[:n_classes//2, :].sum(axis=0)
    pos_of_each_prediction = cls_confusion_mat[n_classes//2 + 1:, :].sum(axis=0)

    largest_number_digit_len += 1

    matrix_output += '\n'
    matrix_output += 'Predicted\t'
    for label in range(n_classes):
        matrix_output += f'{label:>{largest_number_digit_len}}\t'
    matrix_output += '\n'

    matrix_output +='neg in preds:\t'
    for i in range(n_classes):
        matrix_output += f'{neg_of_each_prediction[i]:>{largest_number_digit_len}}\t'
    matrix_output += '\n'
    matrix_output += 'pos in preds:\t'
    for i in range(n_classes):
        matrix_output +=f'{pos_of_each_prediction[i]:>{largest_number_digit_len}}\t'
    matrix_output += '\n'

    print_log(matrix_output, level='INFO')

    precision_of_neg = neg_neg / (neg_neg + pos_neg) if (neg_neg + pos_neg) > 0 else 0
    precision_of_pos = pos_pos / (pos_pos + neg_pos) if (pos_pos + neg_pos) > 0 else 0

    return precision_of_neg, precision_of_pos