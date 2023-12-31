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
from tensorflow.keras.initializers import GlorotNormal

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

def create_compiled_model(window_size = window_length, feature_dim = 14, kernel_size = (10, 1), filter_num = 10, dropout_rate = 0.5):
    model = tf.keras.Sequential([
        layers.Conv2D(filters = filter_num, kernel_size = kernel_size, padding ='same', activation ='tanh', kernel_initializer =GlorotNormal(), input_shape = (window_size, feature_dim, 1)),
        layers.Conv2D(filters = filter_num, kernel_size = kernel_size, padding ='same', activation ='tanh', kernel_initializer =GlorotNormal()),
        layers.Conv2D(filters = filter_num, kernel_size = kernel_size, padding ='same', activation ='tanh', kernel_initializer =GlorotNormal()),
        layers.Conv2D(filters = filter_num, kernel_size = kernel_size, padding ='same', activation ='tanh', kernel_initializer =GlorotNormal()),
        layers.Conv2D(filters = 1, kernel_size = (3, 1), padding = 'same', activation= 'tanh', kernel_initializer = GlorotNormal()),
        layers.Flatten(),
        layers.Dropout(rate = dropout_rate),
        layers.Dense(100, activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Dense(1, kernel_initializer = GlorotNormal())
    ])
    return model

def scheduler(epoch):
  if epoch <= 100:
    return 1e-3
  else:
    return 1e-4

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
CNN = create_compiled_model()
CNN.compile(optimizer = Adam(learning_rate = 1e-3), loss = MeanSquaredError(), metrics = [RootMeanSquaredError()])
DCNN_hist = CNN.fit(x = X_train, y = y_train, shuffle = False, batch_size = 512, epochs = 150,  callbacks = callback, verbose = 2)
print("evaluation of CNN:", CNN.evaluate(X_test, y_test, verbose = 2))
print("CNN fitting finished!")
test_rmse = CNN.evaluate(X_test, y_test, verbose = 2)[1]

y_hat_train = CNN.predict(X_train)
y_hat_calibration = CNN.predict(X_calibration)
y_hat_test = CNN.predict(X_test)
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
                          "CNN-NSCPN"+" intervals")
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCP intervals ","CNN-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCP intervals ","CNN-SCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCPN intervals ","CNN-NSCP intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCP intervals ","CNN-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-NSCP intervals ","CNN-NSCPN intervals","Overlapped intervals"])

plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCPN"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Groundtruth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCPN intervals ","CNN-NSCPN intervals","Overlapped intervals"])

coverage = mf.calculate_coverage(y_test, total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
average_length = mf.calculate_average_length(total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
function_score = mf.calculate_function_score(y_test, y_hat_test)

print("function_score:", function_score)
print("RMSE:", test_rmse)
print("coverage:", coverage)
print("average_length:", average_length)
