import numpy as np
import json

from train_daily_data.global_logger import print_log


def post_process(equity_curve_2d, num_trade_days_to_truncate=0):

    log_base = 1.01

    # convert equity_curve to numpy array
    equity_curve_2d = np.array(equity_curve_2d, dtype=float)

    if num_trade_days_to_truncate > 0 and equity_curve_2d.shape[1] > num_trade_days_to_truncate:
        equity_curve_2d = equity_curve_2d[:, :-num_trade_days_to_truncate]

    # normalize equity by dividing by the initial value
    equity_curve_2d /= equity_curve_2d[:, 0:1]

    # replace NaN and inf with 1e10
    equity_curve_2d[~np.isfinite(equity_curve_2d)] = 1e10
    # replace extremely large values by 1e10
    equity_curve_2d[equity_curve_2d > 1e10] = 1e10
    # replace extremely small values by 1e-10
    equity_curve_2d[equity_curve_2d < 1e-10] = 1e-10

    # convert equity to logarithm format log1.01
    equity_curve_2d = np.log(equity_curve_2d) / np.log(log_base)

    # 2d to 1d 
    equity_curve = np.mean(equity_curve_2d, axis=0)

    return equity_curve


def evaluate(equity_of_log_format, daily_return_half_life):
    """
    equity_curve: list or array of portfolio values (daily total property).
    """
    returns = np.diff(equity_of_log_format)

    # Basic metrics
    # Calculate decay of each day via how many days for the half decay of daily return
    decay_of_each_day = 0.5 ** (1 / daily_return_half_life)
    weights = decay_of_each_day ** np.arange(len(returns)-1, -1, -1)
    # normalize weights
    weights /= np.sum(weights)
    # calculate weighted returns
    weighted_returns = returns * weights
    # calculate weighted mean
    mean_return = np.sum(weighted_returns)

    # Calculate volatility
    volatility = np.std(weighted_returns)

    # Drawdown
    rolling_max = np.maximum.accumulate(equity_of_log_format)
    drawdowns = equity_of_log_format - rolling_max
    # Calculate weighted mean drawdown
    weighted_drawdowns = drawdowns[1:] * weights
    mean_drawdown = np.sum(weighted_drawdowns)

    num_weeks = len(returns) // 5
    num_months = len(returns) // 22
    returns_grouped_by_week = returns[-num_weeks*5:].reshape(num_weeks, 5)
    returns_grouped_by_month = returns[-num_months*22:].reshape(num_months, 22)
    
    # Calculate weekly and monthly returns
    weekly_returns = returns_grouped_by_week.sum(axis=1)
    monthly_returns = returns_grouped_by_month.sum(axis=1)

    # Calculate std of daily returns of each week or each month
    weekly_std = np.std(returns_grouped_by_week, axis=1)
    monthly_std = np.std(returns_grouped_by_month, axis=1)

    assert len(weekly_returns) == len(weekly_std)
    assert len(monthly_returns) == len(monthly_std)

    # Calculate information ratio
    weekly_IR = weekly_returns / (weekly_std + 1e-20)
    monthly_IR = monthly_returns / (monthly_std + 1e-20)

    # Calculate win rate
    weekly_win_rate = np.sum(weekly_returns > 0) / len(weekly_returns) if len(weekly_returns) > 0 else 0
    monthly_win_rate = np.sum(monthly_returns > 0) / len(monthly_returns) if len(monthly_returns) > 0 else 0

    # Calculate medians of weekly_IR and monthly_IR
    weekly_IR_median = np.median(weekly_IR) if len(weekly_IR) > 0 else 0
    monthly_IR_median = np.median(monthly_IR) if len(monthly_IR) > 0 else 0

    return {
        "Mean Return": mean_return,
        "Volatility of Returns": volatility,
        "Mean Drawdown": mean_drawdown,
        "Weekly Win Rate": weekly_win_rate,
        "Monthly Win Rate": monthly_win_rate,
        "Weekly IR Median": weekly_IR_median,
        "Monthly IR Median": monthly_IR_median
    }


def marking_metric(
        evaluation_results,
        volatility_rate,
        drawdown_rate,
        min_weekly_win_rate,
        min_monthly_win_rate,
        min_weekly_IR_median,
        min_monthly_IR_median,
        logging=True):
    """
    Marking metrics for the evaluation results.
    """
    if logging:
        print_log(f"Evaluation Results: {json.dumps(evaluation_results, indent=4)}", level='INFO')

    mean_return = evaluation_results["Mean Return"]
    volatility = evaluation_results["Volatility of Returns"]
    mean_drawdown = evaluation_results["Mean Drawdown"]
    weekly_win_rate = evaluation_results["Weekly Win Rate"]
    monthly_win_rate = evaluation_results["Monthly Win Rate"]
    weekly_IR_median = evaluation_results["Weekly IR Median"]
    monthly_IR_median = evaluation_results["Monthly IR Median"]

    if min_weekly_win_rate > weekly_win_rate:
        return 1e10 * (-1)

    if min_monthly_win_rate > monthly_win_rate:
        return 1e10 * (-1)

    if min_weekly_IR_median > weekly_IR_median:
        return 1e10 * (-1)

    if min_monthly_IR_median > monthly_IR_median:
        return 1e10 * (-1)

    if 1.0 - volatility_rate - drawdown_rate < 0:
        print_log("Warning: The sum of volatility_rate and drawdown_rate exceeds 1.0", level='WARNING')

    mark = mean_return * (1.0 - volatility_rate - drawdown_rate) + \
        volatility * (-1) * volatility_rate + mean_drawdown * drawdown_rate

    if logging:
        print_log(f"Marking Score: {mark}", level='INFO')

    return mark


def evaluate_equity_curve(
        equity_curve,
        daily_return_half_life,
        volatility_rate,
        drawdown_rate,
        min_weekly_win_rate,
        min_monthly_win_rate,
        min_weekly_IR_median,
        min_monthly_IR_median,
        num_trade_days_to_truncate=0,
        logging=True):

    # Preprocess the equity curve into log format
    processed_equity = post_process(equity_curve, num_trade_days_to_truncate)
    # Evaluate the processed equity curve
    evaluation_results = evaluate(processed_equity, daily_return_half_life)
    # Compute the marking score
    marking_score = marking_metric(
        evaluation_results,
        volatility_rate,
        drawdown_rate,
        min_weekly_win_rate,
        min_monthly_win_rate,
        min_weekly_IR_median,
        min_monthly_IR_median,
        logging=logging
    )
    return marking_score, evaluation_results


def most_distinct_candidates(equity_of_log_format_2d, num_to_select):
    """
    Find the most distinct candidates from a 2D array of equity curves.
    """
    # if there is only one candidate, return it directly
    if len(equity_of_log_format_2d) == 1:
        return [0]

    # Make sure the input data is a pure numpy array
    equity_of_log_format_2d = np.array(equity_of_log_format_2d, dtype=float)

    # diff on the second axis
    daily_returns_2d = np.diff(equity_of_log_format_2d, axis=1)
    # Make sure the number of days is multiple of 5
    num_days = daily_returns_2d.shape[1]
    num_days -= num_days % 5
    daily_returns_2d = daily_returns_2d[:, -num_days :]
    # Reshape from (N, T) to (N, T // 5, 5) and then sum along the last axis
    weekly_returns_2d = daily_returns_2d.reshape(daily_returns_2d.shape[0], -1, 5).sum(axis=-1)

    # calculate covariance of weekly returns
    covariance_matrix = np.cov(weekly_returns_2d)

    # Select candidates of least covariance between each other
    pairs_as_dict = dict()
    # Populate the dictionary with pairs and their covariance
    for i in range(covariance_matrix.shape[0]):
        for j in range(i + 1, covariance_matrix.shape[1]):
            pairs_as_dict[(i, j)] = covariance_matrix[i, j]

    # Select pairs with the least covariance
    pairs_sorted = sorted(pairs_as_dict.items(), key=lambda x: x[1])
    # Get indices from selected pairs
    selected_candidates = set()
    for (idx1, idx2), val in pairs_sorted:
        selected_candidates.add(idx1)
        if len(selected_candidates) >= num_to_select:
            break
        selected_candidates.add(idx2)
        if len(selected_candidates) >= num_to_select:
            break

    selected_candidates = sorted(list(selected_candidates))

    return selected_candidates


def select_distinct_candidates(equity_curve_list, num_to_select, num_trade_days_to_tuncate):

    processed_equity_curve_list = []
    for equity_curve in equity_curve_list:
        processed_equity = post_process(equity_curve, num_trade_days_to_tuncate)
        processed_equity_curve_list.append(processed_equity)

    selected_candidates = most_distinct_candidates(processed_equity_curve_list, num_to_select)

    return selected_candidates