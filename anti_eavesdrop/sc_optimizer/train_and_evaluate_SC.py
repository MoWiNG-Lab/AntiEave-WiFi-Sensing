import copy
import numpy as np

from sacred.utils import apply_backspaces_and_linefeeds
from sacred import Experiment as SacredExperiment

from anti_eavesdrop.sc_optimizer.Batch3DTo4DSequence import Batch3DTo4DSequence
from anti_eavesdrop.sc_optimizer.SCModel import SCModel
from anti_eavesdrop.sc_optimizer.get_data_by_schedule import load_data
from ml.train_and_evaluate import add_mongo_db_observer

ex = SacredExperiment(additional_cli_options=[add_mongo_db_observer])
ex.captured_out_filter = apply_backspaces_and_linefeeds


class TrainAndEvaluate:
    def run(self, dataset_params, experiment_params, model_params, mac_params, _run, _log):
        # Setup
        assert(len(experiment_params['S']['n_epochs']) == len(experiment_params['C']['n_epochs']))
        num_rounds = len(experiment_params['S']['n_epochs'])

        # Initialize Models
        SC = SCModel.create_SC(model_params)
        SC.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        batch_r = np.random.random([model_params['S']['n_batches'], 1])
        batch_r_hat = np.random.random([10, 1])

        # Initialize data samples
        print("Initialize data samples")
        X_list, y_list, meta_list, y_columns, all_y_columns = load_data(dataset_params, experiment_params, mac_params, _log)
        X_all = np.array(X_list)
        y = y_list[0]
        meta = meta_list[0]

        # Initialize csi train_sequence
        print("train_sequence")
        train_sequence = Batch3DTo4DSequence(
            X_all, y, meta, y_columns, all_y_columns,
            batch_r=batch_r,
            num_timesteps=dataset_params['num_timesteps'],
            batch_size=model_params['S']['n_batches'],
            should_shuffle=True,
            use_training=True,
            use_testing=False,
        )

        # Initialize csi train_sequence
        print("test_sequence")
        test_sequence = Batch3DTo4DSequence(
            X_all, y, meta, y_columns, all_y_columns,
            batch_r=batch_r,
            num_timesteps=dataset_params['num_timesteps'],
            batch_size=model_params['S']['n_batches'],
            should_shuffle=True,
            use_training=False,
            use_testing=True,
        )

        # Train Models
        for r in range(num_rounds):
            _log.info(f"\n\n\nRound: {r}")

            print(f"\n\n\nTrain C for {experiment_params['C']['n_epochs'][r]} epochs")
            SC = SCModel.train_C(SC, dataset_params, experiment_params, model_params, train_sequence, test_sequence,
                                 epochs=experiment_params['C']['n_epochs'][r], _round=r, _run=_run)

        # Evaluate
        SC.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        return {
            'C': SCModel.evaluate_C(SC, train_sequence, test_sequence),
        }

@ex.config
def my_config():
    model_params = {
        'S': {
            'num_stations': 5,
            'window_size': 20,
            'n_batches': 64,
            'hidden_size': 100,
            'num_hidden_dense_layers': 2,
            'optimizer': 'sgd',
            'loss_weight': 1.0,
            'regularization_losses': {}
        },
        'C': {
            'num_subcarriers': 64,
            'window_size': 20,
            'hidden_size': 25,
            'num_hidden_dense_layers': 2,
            'use_final_dropout': True,
            'num_classes': 5,
            'learning_rate': 0.01,
            'dropout': 0.5,
            'use_kernel_regularization': True,
            'use_activity_regularization': False,
            'optimizer': 'sgd',
            'activation': 'relu',
        }
    }
    dataset_params = {
        "dataset_directory": f"dataset_which_does_not_exist",
        "dataset_name": "Experiment Name",
        'selected_column': 'csi_amplitude',
        'num_timesteps': 20,
        'use_wavelet_denoising': False,
        'actions_to_filter': ['none'],
        'percentage_middle_to_keep': 1.0,
    }
    experiment_params = {
        'epochs': 100,
        'test_size': 0.5,
        'save_plot_every_n_epochs': 10,
        'experiment_group_label': "standard-experiment",
        'S': {
            'n_epochs': [],
        },
        'C': {
            'n_epochs': [],
        },
    }
    mac_params = {
        'train_mac': None,
        'evaluate_mac_list': [],
    }

@ex.automain
def auto_main(_run, _seed, _log, model_params, dataset_params, experiment_params, mac_params):
    dataset_params = copy.deepcopy(dataset_params)
    mac_params = copy.deepcopy(mac_params)
    print("mac_params", mac_params)
    _run.results = TrainAndEvaluate().run(dataset_params, experiment_params, model_params, mac_params, _run, _log)
    return _run.results
