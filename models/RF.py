import random
import numpy as np
import utils.data_process as dp
import utils.metrics_functions as mf
import utils.plot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

random_state = 99
random.seed(random_state)
np.random.seed(random_state)
window_length = 1

dataframe_dictionary = dp.get_dataset_from_txt()
split_dataset_dictionary, train_index, calibration_index = dp.split_dataset(dataframe_dictionary, groups = dataframe_dictionary["train"]["unit"], n_splits = 1, test_size = 0.2, random_state = random_state)
X_train, y_train = dp.get_train_data(split_dataset_dictionary['train'], window_length = window_length, shift = 1, index = train_index)
X_calibration, y_calibration = dp.get_train_data(split_dataset_dictionary['calibration'], window_length = window_length, shift = 1, index = calibration_index)
time_index_calibration = y_calibration
X_test, y_test = dp.get_test_data(split_dataset_dictionary['test'], window_length = window_length, shift = 1, num_test_windows = 1)
X_train = np.squeeze(X_train, axis = 1)
X_calibration = np.squeeze(X_calibration, axis = 1)
X_test = np.squeeze(X_test, axis = 1)

rf_model = RandomForestRegressor(n_estimators = 300, max_features = "sqrt",
                                 n_jobs = -1, random_state = random_state)
rf_model.fit(X_train, y_train)
print("rf_model fitting finished!")
test_rmse = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))

y_hat_train = rf_model.predict(X_train)
y_hat_calibration = rf_model.predict(X_calibration)
y_hat_test = rf_model.predict(X_test)
time_index_test = np.around(y_hat_test)
res_train = np.abs(y_hat_train - y_train)


RF = RandomForestRegressor(random_state = random_state)
RF.fit(X_train, res_train)
print("RF fitting finished!")

base_ratio = 0.9
alpha = 0.1
sigma_calibration = RF.predict(X_calibration).reshape(-1, 1)
sigma_test = RF.predict(X_test).reshape(-1, 1)

y_calibration = y_calibration.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
time_index_calibration = time_index_calibration.reshape(-1, 1)
y_hat_calibration = y_hat_calibration.reshape(-1, 1)
y_hat_test = y_hat_test.reshape(-1, 1)
time_index_test = time_index_test.reshape(-1, 1)

total_prediction_results_dictionary = mf.calculate_intervals(sigma_calibration,sigma_test,y_calibration,y_hat_calibration,y_test,y_hat_test,time_index_calibration,time_index_test,alpha,base_ratio)
plt.plot_single_intervals(total_prediction_results_dictionary["intervals"]["NSCPN"],
                          total_prediction_results_dictionary["Single-point RUL predictions"],
                          total_prediction_results_dictionary["Groundtruth RULs"],
                          "pink",
                          "RF-NSCPN"+" intervals")
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-SCP intervals ","RF-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-SCP intervals ","RF-SCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-SCPN intervals ","RF-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-SCP intervals ","RF-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-NSCP intervals ","RF-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["RF-SCPN intervals ","RF-NSCPN intervals","Overlapped intervals"])

coverage = mf.calculate_coverage(y_test, total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
average_length = mf.calculate_average_length(total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
function_score = mf.calculate_function_score(y_test, y_hat_test)

print("function_score:", function_score)
print("RMSE:", test_rmse)
print("coverage:", coverage)
print("average_length:", average_length)
