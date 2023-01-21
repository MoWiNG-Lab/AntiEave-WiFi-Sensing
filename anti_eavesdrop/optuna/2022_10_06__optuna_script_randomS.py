import optuna
import argparse
import json
from random import shuffle
from importlib import reload

from helpers.load import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--google_drive_directory', help='Google Drive Directory for training')
    parser.add_argument('--optuna_name', help='Name of the experiment to store (and load) from optuna')
    parser.add_argument('--n_trials', help='n_trials to run', default=10)
    parser.add_argument('--batch_size', help='Batch Size', default=1000)
    parser.add_argument('--macs', help='mac addresses (list)')

    parser.add_argument('--random_station_percentages', help='', default='[]')
    parser.add_argument('--min_station_percentage', help='', default=0)

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

    print("min_station_percentage: ", int(args.min_station_percentage))
    print("random_station_percentages: ", json.loads(args.random_station_percentages))

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

        min_station_percentage = int(args.min_station_percentage)
        num_stations = len(json.loads(args.macs))
        assert(min_station_percentage * num_stations <= 100)
        stations_to_suggest = list(range(num_stations))
        shuffle(stations_to_suggest)
        random_station_percentages = [0]*num_stations
        remaining = 100 - ((num_stations - 1) * min_station_percentage)
        for i in stations_to_suggest[0:-1]:
            random_station_percentages[i] = trial.suggest_int(f"random_station_percentages__{i}", low=min_station_percentage, high=remaining)
            remaining = remaining - random_station_percentages[i] + min_station_percentage
        random_station_percentages[stations_to_suggest[-1]] = trial.suggest_int(f"random_station_percentages__{stations_to_suggest[-1]}", low=remaining, high=remaining)
        print("random_station_percentages", random_station_percentages, 100 - sum(random_station_percentages))
        random_station_percentages = [r / 100 for r in random_station_percentages]


        num_stations = len(json.loads(args.macs))
        window_size = 50
        r = ex.run(config_updates={
            "model_params": {
                'S': {
                    'use_random_S': True,
                    'use_seeded_random_S': bool(args.use_seeded_random_S),
                    'use_tiled_random_S': bool(args.use_tiled_random_S),
                    'random_station_percentages': random_station_percentages,
                    'num_stations': num_stations,
                    'window_size': window_size,
                    'n_batches': 2**8,
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
                'percentage_middle_to_keep': 0.9,
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
        # sampler=optuna.samplers.RandomSampler()
    )
    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
