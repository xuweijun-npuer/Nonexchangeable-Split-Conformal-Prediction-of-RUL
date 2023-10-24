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
    split_result = GroupShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = random_state)
    train_index, calibration_index = next(split_result.split(dataframe_dictionary["train"], groups = groups))
    train_df = dataframe_dictionary["train"].iloc[train_index]
    calibration_df = dataframe_dictionary["train"].iloc[calibration_index]
    test_df = dataframe_dictionary["test"]
    split_dataset_dictionary = {
        "train": train_df,
        "calibration": calibration_df,
        "test": test_df
    }
    train_unit_index = train_df['unit'].unique()
    calibration_unit_index = calibration_df['unit'].unique()
    return split_dataset_dictionary, train_unit_index, calibration_unit_index

def dataframe_slice(df, window_length, shift):
    target_data = df.iloc[:, -1]
    num_batches = int(np.floor((len(df) - window_length) / shift)) + 1
    df = df.drop(['unit', 'time', 'rul'], axis = 1)
    num_features = df.shape[1]
    output_data = np.repeat(np.nan, repeats = num_batches * window_length * num_features).reshape(num_batches,
                                                                                                window_length, -1)
    output_targets = np.repeat(np.nan, repeats = num_batches)
    for batch in range(num_batches):
        output_data[batch, :, :] = df.iloc[(0 + shift * batch):(0 + shift * batch + window_length), :]
        output_targets[batch] = target_data.iloc[(shift * batch + (window_length - 1))]
    return output_data, output_targets

def test_dataframe_slice(df, window_length, shift, num_test_windows = 1):
    required_len = (num_test_windows - 1) * shift + window_length
    batched_test_data_for_an_engine, output_test = dataframe_slice(df.iloc[-required_len:, :],
                                                                          window_length = window_length, shift=shift)
    return batched_test_data_for_an_engine, output_test
def get_train_data(df, window_length, shift, index):
    processed_train_data = []
    processed_train_targets = []
    for i in index:
        temp_df = df[df['unit'] == i]
        output_for_an_engine, output_target_for_an_engine = dataframe_slice(df = temp_df, window_length = window_length, shift = shift)
        processed_train_data.append(output_for_an_engine)
        processed_train_targets.append(output_target_for_an_engine)
    return np.vstack(processed_train_data), np.concatenate(processed_train_targets)

def get_test_data(df, window_length, shift, num_test_windows):
    processed_test_data = []
    processed_test_targets = []
    num_test_engines = len(df['unit'].unique())
    for i in np.arange(1, num_test_engines + 1):
        temp_df = df[df['unit'] == i]
        output_for_an_engine, output_target_for_an_engine = test_dataframe_slice(df = temp_df, window_length = window_length, shift = shift,
                                                                                 num_test_windows = num_test_windows)
        processed_test_data.append(output_for_an_engine)
        processed_test_targets.append(output_target_for_an_engine)
    return np.vstack(processed_test_data), np.concatenate(processed_test_targets)

