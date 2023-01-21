import numpy as np
import pandas as pd

from preprocessing.transform_dataset import transform_dataset


def load_data(dataset_params, experiment_params, mac_params, _log):
    #
    # Load dataset and split by `evaluate_mac_list`
    #
    X_list = []
    y_list = []
    meta_list = []
    y_columns = []
    all_y_columns = []
    for mac in mac_params['evaluate_mac_list']:
        dataset_params['mac_address'] = mac
        X, y, meta, y_columns, all_y_columns = transform_dataset(dataset_params, experiment_params, _log=_log)
        X_list.append(X)
        y_list.append(y)
        meta_list.append(meta)
        _log.info(f"X.shape for {mac}: {X.shape}")

    #
    # "Join Xs" from each mac_address together into a single `X`
    #
    _log.info(meta_list[0])
    _min = meta_list[0]['interpolated_timestamp'].min()
    _max = meta_list[0]['interpolated_timestamp'].max()
    for _meta in meta_list:
        _min = max(_min, _meta['interpolated_timestamp'].min())
        _max = min(_max, _meta['interpolated_timestamp'].max())

    #
    # Interpolate data based on timestamp
    #
    for i in range(len(meta_list)):
        meta_list[i] = meta_list[i][
            (meta_list[i]['interpolated_timestamp'] >= _min) &
            (meta_list[i]['interpolated_timestamp'] <= _max)
            ]
        X_list[i] = X_list[i].iloc[meta_list[i].index].reset_index(drop=True)
        y_list[i] = y_list[i].iloc[meta_list[i].index].reset_index(drop=True)
        meta_list[i] = meta_list[i].reset_index(drop=True)

    return X_list, y_list, meta_list, y_columns, all_y_columns


def get_data_by_schedule(X_list, y_list, meta_list, schedules):
    full_schedule = []
    while len(full_schedule) < len(X_list[0]):
        for s in schedules:
            full_schedule += s.tolist()
    full_schedule = np.array(full_schedule[0:len(X_list[0])])

    X = pd.DataFrame(np.zeros(X_list[0].shape))
    for i in range(len(meta_list)):
        X[full_schedule == i] = X_list[i][full_schedule == i]

    y = y_list[0]
    meta = meta_list[0]

    return X, y, meta
