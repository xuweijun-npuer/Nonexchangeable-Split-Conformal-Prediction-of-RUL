import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

useless_columns = ['sm1', 'sm5', 'sm6', 'sm10', 'sm16', 'sm18', 'sm19']
ignorable_columns = ['os1', 'os2', 'os3']
drop_columns = useless_columns + ignorable_columns
temporary_drop_columns = ['unit','time']
all_drop_columns = temporary_drop_columns + drop_columns

def get_dataset_from_txt():
    column_names = []
    column_names.append('unit')
    column_names.append('time')
    for i in range(1, 4):
        column_names.append('os' + str(i))
    for i in range(1, 22):
        column_names.append('sm' + str(i))

    train_dataframe = pd.read_csv("../data/train_FD001.txt", header = None, delim_whitespace = True)
    test_dataframe = pd.read_csv("../data/test_FD001.txt", header = None, delim_whitespace = True)
    rul_dataframe = pd.read_csv("../data/RUL_FD001.txt", header = None, delim_whitespace = True)

    train_dataframe.columns = column_names
    test_dataframe.columns = column_names

    train_dataframe, test_dataframe = normalize_dataframe(train_dataframe, test_dataframe)

    train_dataframe = add_rul(train_dataframe)
    test_dataframe = add_rul(test_dataframe, rul_dataframe)

    train_dataframe = piecewise_rul(train_dataframe)
    test_dataframe = piecewise_rul(test_dataframe)

    dataframe_dictionary = {
        "train": train_dataframe,
        "test": test_dataframe
    }
    return dataframe_dictionary

def normalize_dataframe(train_dataframe, test_dataframe):
    scaler = StandardScaler()
    temporary_train_data = scaler.fit_transform(train_dataframe.drop(all_drop_columns, axis = 1))
    temporary_test_data = scaler.transform(test_dataframe.drop(all_drop_columns, axis = 1))
    temporary_train_df = pd.DataFrame(data = temporary_train_data, index = train_dataframe.index, columns = train_dataframe.drop(all_drop_columns, axis = 1).columns)
    temporary_test_df = pd.DataFrame(data = temporary_test_data, index = test_dataframe.index, columns = test_dataframe.drop(all_drop_columns, axis = 1).columns)
    normalized_train_dataframe = pd.concat([train_dataframe['unit'], train_dataframe['time'], temporary_train_df], axis = 1)
    normalized_test_dataframe = pd.concat([test_dataframe['unit'], test_dataframe['time'], temporary_test_df], axis = 1)
    return normalized_train_dataframe, normalized_test_dataframe

def add_rul(df, rul_df = None):
    rul = []
    for i in df['unit'].unique():
        time_list = df[df['unit'] == i]['time'].values
        length = len(time_list)
        if rul_df is None:
            rul += list(length - time_list)
        else:
            rul_val = rul_df.iloc[i - 1].item()
            rul += list(length + rul_val - time_list)
    df['rul'] = rul
    return df

def piecewise_rul(df):
    df_copy = df.copy()
    df_copy['rul'] = df_copy['rul'].apply(lambda x: 125 if x > 125 else x)
    return df_copy

def split_dataset(dataframe_dictionary, groups, n_splits, test_size, random_state):
    split_result = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state = random_state)
    train_index, calibration_index = next(split_result.split(dataframe_dictionary["train"], groups = groups))
    train_df = dataframe_dictionary["train"].iloc[train_index]
    calibration_df = dataframe_dictionary["train"].iloc[calibration_index]
    test_df = dataframe_dictionary["test"]
    split_dataset_dictionary = {
        "train": train_df,
        "calibration": calibration_df,
        "test": test_df
    }
    return split_dataset_dictionary

def reform_dictionary(df_dictionary, window_length):
    train = reform_dataframe(df_dictionary["train"], number_in = window_length - 1, number_out = 1)
    calibration = reform_dataframe(df_dictionary["calibration"], number_in = window_length - 1, number_out = 1)
    test_temporary = reform_dataframe(df_dictionary["test"], number_in = window_length - 1, number_out=1)
    test = reform_test_dataframe(test_temporary)
    reform_dictionary = {
        "train": train,
        "calibration": calibration,
        "test": test
    }
    return reform_dictionary

def reform_dataframe(dataframe, number_in, number_out):
    X_list, y_list, unit_list, time_index_list = [], [], [], []
    for unit in dataframe.unit.unique():
        unit_dataframe = dataframe[dataframe.unit == unit]
        id_df_supervised = series_to_supervised(unit_dataframe.drop(["unit", "time", "rul"], axis=1), number_in, number_out)
        X = id_df_supervised.astype(np.float32).values
        X_list.append(X.reshape(X.shape[0], number_in+1, X.shape[1]//(number_in+1), 1))
        rul = unit_dataframe["rul"].astype(np.float32).values.reshape(-1,1)
        y_list.append(rul[number_in:])
        unit_list = unit_list + X.shape[0]*[unit]
        time_index_list.append(np.arange(X.shape[0]))

    return {
        "X": np.vstack(X_list),
        "y": np.vstack(y_list),
        "unit": np.array(unit_list),
        "time_index": np.hstack(time_index_list)
    }

def series_to_supervised(dataframe, number_in, number_out):
    number_dim = dataframe.shape[1]
    columns, names = list(), list()
    for i in range(number_in, 0, -1):
        columns.append(dataframe.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(number_dim)]
    for i in range(0, number_out):
        columns.append(dataframe.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(number_dim)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(number_dim)]
    new_dataframe = pd.concat(columns, axis=1)
    new_dataframe.columns = names
    new_dataframe.dropna(inplace=True)
    return new_dataframe

def reform_test_dataframe(test_dataframe):
    X_test, y_test, time_index_test = [], [], []
    for unit in np.unique(test_dataframe["unit"]):
        X_test.append(test_dataframe["X"][test_dataframe["unit"]==unit][-1])
        y_test.append(test_dataframe["y"][test_dataframe["unit"]==unit][-1])
        time_index_test.append(test_dataframe["time_index"][test_dataframe["unit"]==unit][-1])
    return {
        "X":np.array(X_test),
        "y":np.array(y_test),
        "time_index":np.array(time_index_test)
    }

def prepare_data_for_multimodel(df_dictionary):
    X_train = df_dictionary["train"]["X"]
    y_train = df_dictionary["train"]["y"]
    X_calibration = df_dictionary["calibration"]["X"]
    y_calibration = df_dictionary["calibration"]["y"]
    time_index_calibration = df_dictionary["calibration"]["time_index"]
    X_test = df_dictionary["test"]["X"]
    y_test = df_dictionary["test"]["y"]
    time_index_test = df_dictionary["test"]["time_index"]
    data_dictionary = {
        "X_train": X_train,
        "X_calibration": X_calibration,
        "X_test": X_test,
        "y_train": y_train,
        "y_calibration": y_calibration,
        "y_test": y_test,
        "time_index_calibration": time_index_calibration,
        "time_index_test": time_index_test
    }
    return data_dictionary


