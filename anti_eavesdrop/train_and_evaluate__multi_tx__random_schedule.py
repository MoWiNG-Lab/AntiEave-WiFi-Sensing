#!/usr/bin/python3

import copy
import time
import os
import sys
import shutil
import json
import pandas as pd

sys.path.append(os.path.dirname(__file__) + "/../../")

import numpy as np

from sacred import Experiment as SacredExperiment, cli_option
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from helpers.storage import tmp_dir
from helpers.config import config as c
from ml.train_and_evaluate import TrainAndEvaluate
from ml.Batch2DTo3DSequence import Batch2DTo3DSequence

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@cli_option('-o', '--omniboard_config', is_flag=False)
def add_mongo_db_observer(args, run):
    url = c("omniboard", args + ".mongodbURI")
    db_name = c("omniboard", args + ".path").replace("/", "")
    run.observers.append(MongoObserver(url=url, db_name=db_name))

ex = SacredExperiment(additional_cli_options=[add_mongo_db_observer])
ex.captured_out_filter = apply_backspaces_and_linefeeds


class TrainAndEvaluateMultiTXRandomSchedule(TrainAndEvaluate):
    def bash(self, config, python_script="./projects/anti_eavesdrop/train_and_evaluate__multi_tx__random_schedule.py", run_in_background=False, omniboard_config=None, tsp=False):
        """
        Print shell command string to initialize training at the terminal
        :param config: dictionary containing experiment parameters
        :param python_script: relative location of python script (we assume that terminal is pointed to `cd ~/experiments_server/python`)
        :param run_in_background:
        :return:
        """

        if omniboard_config is None:
            raise Exception("Please pass in `omniboard_config` string to store results!")

        out = "tsp " if tsp else ""
        out += f'python {python_script} --omniboard_config {omniboard_config} with'
        for k in list(config.keys()):
            x = json.dumps(config[k]).replace('"', "'").replace("true", "True").replace("false", "False")
            out += ' ' + k + f'=\"{x}\"'
        if run_in_background:
            out += " > /dev/null 2>&1 &"

        return out + "\n\n"

    def tsp(self, config, python_script="../projects/anti_eavesdrop/train_and_evaluate__multi_tx__random_schedule.py", omniboard_config=None):
        """
        Print tsp command string for use with the task-spooler unix utility
        :param config:
        :return:
        """

        if omniboard_config is None:
            raise Exception("Please pass in `omniboard_config` string to store results!")

        def filter_nulls(d):
            if type(d) is not dict:
                return d
            return {k: filter_nulls(v) for k, v in d.items() if v is not None}

        out = f'tsp docker-compose exec -T jupyter ipython -c "%run {python_script} --omniboard_config {omniboard_config} with'
        for k in list(config.keys()):
            x = json.dumps(filter_nulls(config[k])).replace('"', "'").replace("true", "True").replace("false", "False")
            out += ' ' + k + f'=\\"{x}\\"'
        out += ' --debug"\n'
        return out + "\n"

    def get_train_X(self, dataset_params, experiment_params, mac_params, _log):
        dataset_params['mac_address'] = mac_params['train_mac']
        return self.transform_dataset(dataset_params, experiment_params, _log=_log)

    def get_evaluate_X(self, dataset_params, experiment_params, mac_params, _log):
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
            X, y, meta, y_columns, all_y_columns = self.transform_dataset(dataset_params, experiment_params, _log=_log)
            X_list.append(X)
            y_list.append(y)
            meta_list.append(meta)
            _log.info(f"X.shape for {mac}: {X.shape}")

        #
        # "Join Xs" from each mac_address together into a single `X`
        #
        _min = meta_list[0]['interpolated_timestamp'].min()
        _max = meta_list[0]['interpolated_timestamp'].max()
        for _meta in meta_list:
            _min = max(_min, _meta['interpolated_timestamp'].min())
            _max = min(_max, _meta['interpolated_timestamp'].max())
            print(_meta['interpolated_timestamp'].min(), _meta['interpolated_timestamp'].max())

        for i in range(len(meta_list)):
            meta_list[i] = meta_list[i][
                (meta_list[i]['interpolated_timestamp'] >= _min) &
                (meta_list[i]['interpolated_timestamp'] <= _max)
                ]
            X_list[i] = X_list[i].iloc[meta_list[i].index].reset_index(drop=True)
            y_list[i] = y_list[i].iloc[meta_list[i].index].reset_index(drop=True)
            meta_list[i] = meta_list[i].reset_index(drop=True)

        random_schedule = np.floor(np.random.rand(len(X_list[0])) * len(mac_params['evaluate_mac_list']))

        X = pd.DataFrame(np.zeros(X_list[0].shape))
        for i in range(len(meta_list)):
            X[random_schedule == i] = X_list[i][random_schedule == i]

        y = y_list[0]
        meta = meta_list[0]
        _log.info(f"X.shape after joining macs: {X.shape}")
        #
        # End "Join Xs".
        # Everything else from here should be exactly like the normal `train_and_evaluate`.
        #
        print("X.index", X.index)
        print("y.index", y.index)
        print("meta.index", meta.index)

        return X, y, meta, y_columns, all_y_columns

    @ex.capture
    def run(self, _run, _seed, _log, config):
        config = copy.deepcopy(config)
        model_params = config['model_params']
        dataset_params = config['dataset_params']
        experiment_params = config['experiment_params']
        mac_params = config['mac_params']

        X, y, meta, y_columns, all_y_columns = self.get_train_X(dataset_params, experiment_params, mac_params, _log)

        train_sequence = Batch2DTo3DSequence(
            X, y, meta, y_columns, all_y_columns,
            num_timesteps=dataset_params['num_timesteps'],
            batch_size=dataset_params['batch_size'],
            should_shuffle=True,
            use_training=True,
            use_testing=False,
        )

        X, y, meta, y_columns, all_y_columns = self.get_evaluate_X(dataset_params, experiment_params, mac_params, _log)

        test_sequence = Batch2DTo3DSequence(
            X, y, meta, y_columns, all_y_columns,
            num_timesteps=dataset_params['num_timesteps'],
            batch_size=dataset_params['batch_size'],
            should_shuffle=False,
            use_training=False,
            use_testing=True,
        )

        train_and_test_sequence = None

        model_params['num_classes'] = len(y_columns)
        model_params['input_dim'] = X.shape[1]

        train_metrics, test_metrics = self.train(_run, _seed, train_sequence, test_sequence, train_and_test_sequence, model_params, dataset_params, experiment_params)

        return train_metrics, test_metrics

@ex.config
def my_config():
    model_params = {
        'csi_window_size': 100,
        'learning_rate': 0.01,
        'hidden_size': 25,
        'num_hidden_dense_layers': 2,
        'dropout': 0.5,
        'use_final_dropout': True,
        'use_kernel_regularization': True,
        'use_activity_regularization': True,
        'optimizer': 'sgd',
    }
    dataset_params = {
        "dataset_directory": f"{tmp_dir()}/dataset_which_does_not_exist",
        "dataset_name": "Experiment Name",
        'selected_column': 'csi_amplitude',
        'transform__diff': False,
        'psd': False,
        'num_timesteps': 100,
        'use_wavelet_denoising': False,
        'pca': 0,
        'ica': 0,
        'only_use_every_nth_frame': 1,
        'actions_to_filter': ['none'],
        'percentage_middle_to_keep': 1.0,
        'equalize_samples_per_class': False,
        'train_only_use_every_nth_frame': 1,
        'test_only_use_every_nth_frame': 1,
        'train_frame_probability': 1,
        'test_frame_probability': 1,
        'time_series_stratified': False,
        'keep_first_n_action_segments': -1,
        'keep_first_n_training_action_segments': -1,
        'ignore_action_segment_by_index': [],
        'rename_actions': None,
        'set_full_column_list': None,
        'batch_size': None,
    }
    experiment_params = {
        'epochs': 100,
        'test_size': 0.5,
        'save_plot_every_n_epochs': 10,
        'experiment_group_label': "standard-experiment",  # Use when you want to group multiple experiments together for later analysis
    }
    mac_params = {
        'train_mac': None,
        'evaluate_mac_list': [],
    }

@ex.automain
def auto_main(_run, _seed, _log, model_params, dataset_params, experiment_params, mac_params):
    if len(_run.observers) == 0:
        raise Exception("No sacred observers found. Did you forget to use `--omniboard_config='default'`?")

    config = {
        "model_params": model_params,
        "dataset_params": dataset_params,
        "experiment_params": experiment_params,
        'mac_params': mac_params,
    }

    _seed = str(_seed) + "--" + str(time.time())

    train_metrics, test_metrics = TrainAndEvaluateMultiTXRandomSchedule().run(_run, _seed, _log, config)

    _run.results = {"train_metrics": train_metrics, "test_metrics": test_metrics}

    # Remove tmp directory (and files) from the experiment
    shutil.rmtree(f"{tmp_dir()}/{_seed}/", ignore_errors=True)
