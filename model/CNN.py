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
split_dataset_dictionary = dp.split_dataset(dataframe_dictionary, groups = dataframe_dictionary["train"]["unit"], n_splits = 1, test_size = 0.2, random_state = random_state)
reform_dictionary = dp.reform_dictionary(split_dataset_dictionary, window_length)
data_dictionary = dp.prepare_data_for_multimodel(reform_dictionary)

X_train = data_dictionary["X_train"]
y_train = data_dictionary["y_train"]
X_calibration = data_dictionary["X_calibration"]
y_calibration = data_dictionary["y_calibration"]
time_index_calibration = data_dictionary["time_index_calibration"]
X_test = data_dictionary["X_test"]
y_test = data_dictionary["y_test"]
time_index_test = data_dictionary["time_index_test"]

def create_compiled_model(window_size=30, feature_dim=14, kernel_size=(10, 1), filter_num=10, dropout_rate=0.5):
    model = tf.keras.Sequential([
        layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal(), input_shape=(window_size, feature_dim, 1)),
        layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Conv2D(filters=filter_num, kernel_size=kernel_size, padding='same', activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Flatten(),
        layers.Dropout(rate=dropout_rate),
        layers.Dense(100, activation='tanh', kernel_initializer=GlorotNormal()),
        layers.Dense(1, kernel_initializer=GlorotNormal())
    ])
    return model

def scheduler(epoch):
  if epoch <= 100:
    return 1e-3
  else:
    return 1e-4

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
CNN = create_compiled_model()
CNN.compile(optimizer=Adam(learning_rate=1e-3), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
DCNN_hist = CNN.fit(x=X_train, y=y_train, shuffle=False, batch_size = 512, epochs = 150,  callbacks=callback, verbose=2)
print("evaluation of CNN:", CNN.evaluate(X_test, y_test, verbose=2))
print("CNN fitting finished!")
test_rmse = CNN.evaluate(X_test, y_test, verbose=2)[1]

y_hat_train = CNN.predict(x = X_train, verbose=0)
y_hat_calibration = CNN.predict(x = X_calibration, verbose=0)
y_hat_test = CNN.predict(x = X_test, verbose=0)
res_train = np.abs(y_hat_train - y_train)

X_train_reshaped = X_train.reshape(-1, 30*14)
X_cal_reshaped = X_calibration.reshape(-1, 30*14)
X_test_reshaped = X_test.reshape(-1, 30*14)

res_train = res_train.ravel()
RF = RandomForestRegressor(random_state=random_state)
RF.fit(X_train_reshaped, res_train)
print("RF fitting finished!")

base_ratio = 0.99
alpha = 0.1
sigma_calibration = RF.predict(X_cal_reshaped).reshape(-1, 1)
sigma_test = RF.predict(X_test_reshaped).reshape(-1, 1)

total_prediction_results_dictionary = mf.calculate_intervals(sigma_calibration,sigma_test,y_calibration,y_hat_calibration,y_test,y_hat_test,time_index_calibration,time_index_test,alpha,base_ratio,window_length)
plt.plot_single_intervals(total_prediction_results_dictionary["intervals"]["NSCPN"],
                          total_prediction_results_dictionary["Single-point RUL predictions"],
                          total_prediction_results_dictionary["Ground truth RULs"],
                          "palegreen",
                          "CNN-NSCPN"+" intervals")
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCP"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Ground truth RULs"],
                       color = ["silver","pink"],
                       label = ["CNN-SCP intervals ","CNN-NSCP intervals","Overlapped intervals"])
plt.plot_two_intervals(total_prediction_results_dictionary["intervals"]["SCP"],
                       total_prediction_results_dictionary["intervals"]["NSCPN"],
                       total_prediction_results_dictionary["Single-point RUL predictions"],
                       total_prediction_results_dictionary["Ground truth RULs"],
                       color=["silver","lightgreen"],
                       label = ["CNN-SCP intervals ","CNN-NSCPN intervals","Overlapped intervals"])
coverage = mf.calculate_coverage(y_test, total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
average_length = mf.calculate_average_length(total_prediction_results_dictionary["intervals"]["NSCPN"][0],total_prediction_results_dictionary["intervals"]["NSCPN"][1])
function_score = mf.calculate_function_score(y_test, y_hat_test)

print("function_score:", function_score)
print("RMSE:", test_rmse)
print("coverage:", coverage)
print("average_length:", average_length)