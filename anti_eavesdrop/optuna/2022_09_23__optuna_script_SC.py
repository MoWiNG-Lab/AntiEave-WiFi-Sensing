import optuna
import argparse
import json
from importlib import reload

from helpers.load import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--google_drive_directory', help='Google Drive Directory for training')
    parser.add_argument('--optuna_name', help='Name of the experiment to store (and load) from optuna')
    parser.add_argument('--n_trials', help='n_trials to run', default=10)
    parser.add_argument('--batch_size', help='Batch Size', default=1000)
    parser.add_argument('--macs', help='mac addresses (list)')
    parser.add_argument('--model_to_evaluate', help='Options: S, C', default='C')
    parser.add_argument('--regularization_losses', help='', default='[]')
    parser.add_argument('--use_random_S', help='', default=False)
    parser.add_argument('--use_seeded_random_S', help='', default=False)
    parser.add_argument('--use_tiled_random_S', help='', default=False)

    parser.add_argument('--S__n_epochs', help='(list) Number of epochs used to train S per round.')
    parser.add_argument('--C__n_epochs', help='(list) Number of epochs used to train C per round.')

    args = parser.parse_args()

    print("google_drive_directory: ", args.google_drive_directory)
    print("optuna_name: ", args.optuna_name)
    print("n_trials: ", args.n_trials)
    print("batch_size: ", int(args.batch_size))
    print("macs: ", json.loads(args.macs))
    print("model_to_evaluate: ", args.model_to_evaluate)
    print("regularization_losses: ", json.loads(args.regularization_losses))
    print("use_random_S: ", bool(args.use_random_S))
    print("use_seeded_random_S: ", bool(args.use_seeded_random_S))
    print("use_tiled_random_S: ", bool(args.use_tiled_random_S))

    print("S__n_epochs: ", json.loads(args.S__n_epochs))
    print("C__n_epochs: ", json.loads(args.C__n_epochs))

    def objective(trial):
        global args
        import anti_eavesdrop.sc_optimizer.train_and_evaluate_SC
        reload(anti_eavesdrop.sc_optimizer.train_and_evaluate_SC)
        from sacred.observers import MongoObserver

        from anti_eavesdrop.sc_optimizer.train_and_evaluate_SC import ex
        from helpers.config import config as c

        url = c("omniboard", "SC.mongodbURI")
        db_name = c("omniboard", "SC.path").replace("/", "")
        ex.observers = [MongoObserver(url=url, db_name=db_name)]
        ex.optuna_trial = trial

        selected_experiment = load(args.google_drive_directory)

        num_stations = len(json.loads(args.macs))
        window_size = 50
        r = ex.run(config_updates={
            "model_params": {
                'S': {
                    'use_random_S': bool(args.use_random_S),
                    'use_seeded_random_S': bool(args.use_seeded_random_S),
                    'use_tiled_random_S': bool(args.use_tiled_random_S),
                    'num_stations': num_stations,
                    'window_size': window_size,
                    'n_batches': 2**trial.suggest_int("S__n_batches", low=4, high=10),
                    'hidden_size': trial.suggest_int("S__hidden_size", low=25, high=200, step=25),
                    'num_hidden_dense_layers': trial.suggest_int("S__num_hidden_dense_layers", low=1, high=4),
                    'regularization_losses': json.loads(args.regularization_losses),
                },
                'C': {
                    'num_subcarriers': 64,
                    'window_size': window_size,
                    'hidden_size': 190,
                    'num_hidden_dense_layers': 4,
                    'use_final_dropout': True,
                    'num_classes': 5,
                    'learning_rate': 0.01,
                    'dropout': 0.4,
                    'use_kernel_regularization': True,
                    'use_activity_regularization': False,
                    'optimizer': 'sgd',
                    'activation': 'relu',
                }
            },
            'dataset_params': {
                'dataset_directory': selected_experiment.data_dir,
                'dataset_name': selected_experiment.experiment_name,
                'selected_column': 'csi_amplitude',
                'num_timesteps': window_size,
                'equalize_samples_per_class': True,
                # 'batch_size': int(args.batch_size),
                'keep_first_n_action_segments': 4*8*2, # n_actions,n_repetitions_to_keep,one_action_plus_one_none
                'pca': 64,
                'only_use_every_nth_frame': 1,
                'percentage_middle_to_keep': trial.suggest_float("percentage_middle_to_keep", low=0.5, high=1.0, step=0.05),
                'interpolation': {
                    'size_in_milliseconds': 50,
                    'method': 'nearest',
                },
            },
            'experiment_params': {
                'S': {
                    'n_epochs':  json.loads(args.S__n_epochs),
                },
                'C': {
                    'n_epochs':  json.loads(args.C__n_epochs),
                },
                'experiment_group_label': args.optuna_name,
            },
            'mac_params': {
                'input_mac_list': json.loads(args.macs),
                'evaluate_mac_list': json.loads(args.macs),
            }
        })
        trial.set_user_attr("sacred_id", r._id)

        return r.results['C']['test']['accuracy']


    study = optuna.create_study(
        storage='postgresql://postgres:postgresPW@postgres:5432/optuna',
        study_name=args.optuna_name,
        load_if_exists=True,
        direction='minimize',  # Goal is to reduce classifier accuracy with 'S'
    )
    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
