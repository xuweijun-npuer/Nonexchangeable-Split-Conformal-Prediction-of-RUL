import numpy as np
import utils.data_process as dp
import utils.metrics_functions as mf
import utils.plot as plt
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

random_state = 99
tf.keras.utils.set_random_seed(random_state)
tf.config.experimental.enable_op_determinism()
window_length = 30

dataframe_dictionary = dp.get_dataset_from_txt()
split_dataset_dictionary, train_index, calibration_index = dp.split_dataset(dataframe_dictionary, groups = dataframe_dictionary["train"]["unit"], n_splits = 1, test_size = 0.2, random_state = random_state)
X_train, y_train = dp.get_train_data(split_dataset_dictionary['train'], window_length = window_length, shift = 1, index = train_index)
X_calibration, y_calibration = dp.get_train_data(split_dataset_dictionary['calibration'], window_length = window_length, shift = 1, index = calibration_index)
time_index_calibration = y_calibration
X_test, y_test = dp.get_test_data(split_dataset_dictionary['test'], window_length = window_length, shift = 1, num_test_windows = 1)

def create_compiled_model():
    model = Sequential([
        layers.Dense(500, input_shape = (30, 14), activation = "tanh"),
        layers.Dense(400, activation = "tanh"),
        layers.Dense(300, activation = "tanh"),
        layers.Dense(100, activation = "tanh"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(96, activation = "relu"),
        layers.Dense(128, activation = "relu"),
        layers.Dense(1)
    ])
    return model

def scheduler(epoch):
  if epoch <= 10:
    return 1e-3
  else:
    return 1e-4

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
DNN = create_compiled_model()
DNN.compile(optimizer = Adam(learning_rate = 1e-3), loss = MeanSquaredError(), metrics = RootMeanSquaredError())
history = DNN.fit(x = X_train, y = y_train, batch_size = 512, epochs = 20,  callbacks = callback, verbose = 2)
print("evaluation of DNN:", DNN.evaluate(X_test, y_test, verbose=2))
print("DNN fitting finished!")
test_rmse = DNN.evaluate(X_test, y_test, verbose = 2)[1]

y_hat_train = DNN.predict(X_train)
y_hat_calibration = DNN.predict(X_calibration)
y_hat_test = DNN.predict(X_test)
time_index_test = np.around(y_hat_test)
y_train = y_train.reshape(-1, 1)
res_train = np.abs(y_hat_train - y_train)

RF = RandomForestRegressor(random_state = random_state)
res_train = res_train.ravel()
X_train = X_train.reshape((X_train.shape[0], -1))
RF.fit(X_train, res_train)
print("RF fitting finished!")

base_ratio = 0.9
alpha = 0.1
sigma_calibration = RF.predict(X_calibration.reshape((X_calibration.shape[0], -1))).reshape(-1, 1)
sigma_test = RF.predict(X_test.reshape((X_test.shape[0], -1))).reshape(-1, 1)

y_calibration = y_calibration.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
time_index_calibration = time_index_calibration.reshape(-1, 1)


total_prediction_results_dictionary = mf.calculate_intervals(sigma_calibration,sigma_test,y_calibration,y_hat_calibration,y_test,y_hat_test,time_index_calibration,time_index_test,alpha,base_ratio)
plt.plot_single_intervals(total_prediction_results_dictionary["intervals"]["NSCPN"],
                          total_prediction_results_dictionary["Single-point RUL predictions"],
                          total_prediction_results_dictionary["Groundtruth RULs"],
                          "pink",
                          "DNN-NSCPN"+" intervals")
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-SCP intervals ","DNN-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-SCP intervals ","DNN-SCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-SCPN intervals ","DNN-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-SCP intervals ","DNN-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-NSCP intervals ","DNN-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["DNN-SCPN intervals ","DNN-NSCPN intervals","Overlapped intervals"])

coverage = mf.calculate_coverage(y_test, total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
average_length = mf.calculate_average_length(total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
function_score = mf.calculate_function_score(y_test, y_hat_test)

print("function_score:", function_score)
print("RMSE:", test_rmse)
print("coverage:", coverage)
print("average_length:", average_length)
