import optuna
import argparse
import json
from importlib import reload

from helpers.load import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--google_drive_directory', help='Google Drive Directory for training')
    parser.add_argument('--optuna_name', help='Name of the experiment to store (and load) from optuna')
    parser.add_argument('--n_trials', help='n_trials to run', default=25)
    parser.add_argument('--batch_size', help='Batch Size', default=1000)
    parser.add_argument('--mac', help='mac address for the given TX')
    parser.add_argument('--actions_to_filter', help='List of actions to ignore', default="[]")
    parser.add_argument('--loss', help='Loss function', default='mse')
    parser.add_argument('--mongo_db', help='mongo_db to store sacred experiment results', default='anti_eavesdrop')

    args = parser.parse_args()

    print("google_drive_directory: ", args.google_drive_directory)
    print("optuna_name: ", args.optuna_name)
    print("n_trials: ", args.n_trials)
    print("batch_size: ", int(args.batch_size))
    print("mac: ", args.mac)
    print("actions_to_filter: ", json.loads(args.actions_to_filter))
    print("loss: ", args.loss)
    print("mongo_db: ", args.mongo_db)

    def objective(trial):
        global args
        import ml.train_and_evaluate
        reload(ml.train_and_evaluate)
        from sacred.observers import MongoObserver

        from ml.train_and_evaluate import ex
        from helpers.config import config as c

        url = c("omniboard", f"{args.mongo_db}.mongodbURI")
        db_name = c("omniboard", f"{args.mongo_db}.path").replace("/", "")
        ex.observers = [MongoObserver(url=url, db_name=db_name)]
        ex.optuna_trial = trial

        csi_window_size = trial.suggest_int("csi_window_size", low=10, high=200, step=10)

        selected_experiment = load(args.google_drive_directory)

        r = ex.run(config_updates={
            "model_params": {
                'csi_window_size': csi_window_size,
                'learning_rate': 10**trial.suggest_int("learning_rate", low=-9, high=1),
                'hidden_size': trial.suggest_int("hidden_size", low=10, high=200, step=10),
                'num_hidden_dense_layers': trial.suggest_int("num_hidden_dense_layers", low=1, high=6),
                'dropout': trial.suggest_float("dropout", low=0.0, high=0.9, step=0.1),
                'use_final_dropout': True,
                'use_kernel_regularization': trial.suggest_categorical('use_kernel_regularization', [True, False]),
                'use_activity_regularization': trial.suggest_categorical('use_activity_regularization', [True, False]),
                'optimizer': 'sgd',
                'loss': args.loss,
            },
            "dataset_params": {
                'dataset_directory': selected_experiment.data_dir,
                'dataset_name': selected_experiment.experiment_name,
                'only_use_every_nth_frame': 1,
                'num_timesteps': csi_window_size,
                'equalize_samples_per_class': True,
                'percentage_middle_to_keep': trial.suggest_float("percentage_middle_to_keep", low=0.5, high=1.0, step=0.05),
                'mac_address': args.mac,
                'batch_size': int(args.batch_size),
                'keep_first_n_action_segments': 4*8*2, # n_actions,n_repetitions_to_keep,one_action_plus_one_none
                'actions_to_filter': json.loads(args.actions_to_filter),
                'pca': 64,
            },
            'experiment_params': {
                'epochs': 100,
                'test_size': 0.5,
                'save_plot_every_n_epochs': 100,
                'experiment_group_label': args.optuna_name,
            },
        })
        trial.set_user_attr("sacred_id", r._id)

        return r.results['test_metrics']['accuracy']


    study = optuna.create_study(
        storage='postgresql://postgres:postgresPW@postgres:5432/optuna',
        study_name=args.optuna_name,
        load_if_exists=True,
        direction='maximize',
    )
    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
