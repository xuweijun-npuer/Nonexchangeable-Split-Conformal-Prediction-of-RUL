import random
import numpy as np
import utils.data_process as dp
import utils.metrics_functions as mf
import utils.plot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

random_state = 99
random.seed(random_state)
np.random.seed(random_state)
window_length = 1

dataframe_dictionary = dp.get_dataset_from_txt()
split_dataset_dictionary = dp.split_dataset(dataframe_dictionary, groups = dataframe_dictionary["train"]["unit"], n_splits = 1, test_size = 0.2, random_state = random_state)
reform_dictionary = dp.reform_dictionary(split_dataset_dictionary, window_length)
data_dictionary = dp.prepare_data_for_multimodel(reform_dictionary)

X_train = data_dictionary["X_train"].reshape(-1,14)
y_train = data_dictionary["y_train"].reshape(-1)
X_calibration = data_dictionary["X_calibration"].reshape(-1, 14)
y_calibration = data_dictionary["y_calibration"].reshape(-1)
time_index_calibration = data_dictionary["time_index_calibration"]
X_test = data_dictionary["X_test"].reshape(-1, 14)
y_test = data_dictionary["y_test"].reshape(-1)
time_index_test = data_dictionary["time_index_test"]

parameters = {"C":[1, 10, 50, 100],
            'epsilon':[1, 5, 10, 50],
             'kernel':["rbf"]}
tuned_svm_reg = GridSearchCV(SVR(),parameters,n_jobs = -1, cv= 10)
tuned_svm_reg.fit(X_train, y_train)
print("tuned_svm_reg.best_params_:", tuned_svm_reg.best_params_)
SVR = tuned_svm_reg.best_estimator_
print("SVR fitting finished!")
test_rmse = np.sqrt(mean_squared_error(y_test, SVR.predict(X_test)))

y_hat_train = SVR.predict(X_train)
y_hat_calibration = SVR.predict(X_calibration)
y_hat_test = SVR.predict(X_test)
res_train = np.abs(y_hat_train - y_train)

RF = RandomForestRegressor(random_state = random_state)
RF.fit(X_train, res_train)
print("RF fitting finished!")

base_ratio = 0.99
alpha = 0.1
sigma_calibration = RF.predict(X_calibration)
sigma_test = RF.predict(X_test)

total_prediction_results_dictionary = mf.calculate_intervals(sigma_calibration,sigma_test,y_calibration,y_hat_calibration,y_test,y_hat_test,time_index_calibration,time_index_test,alpha,base_ratio,window_length)
plt.plot_single_intervals(total_prediction_results_dictionary["intervals"]["NSCPN"],
                          total_prediction_results_dictionary["Single-point RUL predictions"],
                          total_prediction_results_dictionary["Ground truth RULs"],
                          "palegreen",
                          "SVR-NSCPN"+" intervals")
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Ground truth RULs"],
                       color = ["silver","pink"],
                       label = ["SVR-SCP intervals ","SVR-NSCP intervals","Overlapped intervals"])
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Ground truth RULs"],
                       color=["silver","lightgreen"],
                       label = ["SVR-SCP intervals ","SVR-NSCPN intervals","Overlapped intervals"])
coverage = mf.calculate_coverage(y_test, total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
average_length = mf.calculate_average_length(total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
function_score = mf.calculate_function_score(y_test, y_hat_test)

print("function_score:", function_score)
print("RMSE:", test_rmse)
print("coverage:", coverage)
print("average_length:", average_length)