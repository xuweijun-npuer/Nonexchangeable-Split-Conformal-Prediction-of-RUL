import numpy as np

def calculate_function_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13.)-1, np.exp(diff/10.)-1))

def calculate_coverage(rul_true, prediction_lower, prediction_upper):
    number_of_fall_in_range = np.sum((rul_true >= prediction_lower) & (rul_true <= prediction_upper))
    coverage = number_of_fall_in_range/len(rul_true)
    return coverage

def calculate_average_length(prediction_lower, prediction_upper):
    average_length = np.mean(abs(prediction_upper - prediction_lower))
    return average_length

def calculate_quantile(scores, alpha):
    n = len(scores)
    return np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method="inverted_cdf")

def calculate_weight(base_ratio, test_index, calibration_index):
    return base_ratio**np.abs(test_index - calibration_index)

def calculate_quantiles_nonexchangeable(base_ratio, scores, test_index, calibration_index, alpha):
    sorted_scores_index = scores.argsort(axis=0)
    sorted_scores = scores[sorted_scores_index]
    quantiles = []
    positons = []
    for index in test_index:
        weights = calculate_weight(base_ratio, index, calibration_index[sorted_scores_index])
        weights_normalized = weights/(weights.sum()+1)
        position_index = np.where(weights_normalized.cumsum() >= 1-alpha)[0][0]
        quantile = sorted_scores[position_index][0]
        quantiles.append(quantile)
        positons.append(position_index)
    return np.array(quantiles).reshape(-1,1)
def calculate_intervals(sigma_calibration,sigma_test,y_calibration,y_hat_calibration,y_test,y_hat_test,calibration_index,test_index,alpha,base_ratio,window_length):
    scores = np.abs(y_calibration - y_hat_calibration)
    scores_normalized = scores / sigma_calibration
    q = calculate_quantile(scores, alpha)
    if window_length == 30:
        q_array = calculate_quantiles_nonexchangeable(base_ratio, scores, test_index, calibration_index, alpha)
        q_array_normalized = calculate_quantiles_nonexchangeable(base_ratio, scores_normalized, test_index,
                                                                 calibration_index, alpha)
    else:
        q_array = calculate_quantiles_nonexchangeable(base_ratio, scores.reshape(-1, 1), test_index, calibration_index, alpha).reshape(-1)
        q_array_normalized = calculate_quantiles_nonexchangeable(base_ratio, scores_normalized.reshape(-1, 1), test_index,
                                                                 calibration_index, alpha).reshape(-1)
    intervals_dictionary = {
        "SCP": (y_hat_test - q, y_hat_test + q),
        "NSCP": (y_hat_test - q_array, y_hat_test + q_array),
        "NSCPN": (y_hat_test - q_array_normalized * sigma_test, y_hat_test + q_array_normalized * sigma_test),
    }
    total_prediction_results_dictionary ={
        "Ground truth RULs": y_test,
        "Single-point RUL predictions": y_hat_test,
        "intervals": intervals_dictionary
    }
    return total_prediction_results_dictionary
