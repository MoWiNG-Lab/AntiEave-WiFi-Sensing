import numpy as np
import tensorflow as tf
from keras import layers, regularizers
from keras.callbacks import LambdaCallback

from anti_eavesdrop.sc_optimizer.layers import StationSelectionAggregator, \
    ConstantSTALayer, RandomSTALayer


class SCModel:

    @staticmethod
    def find_layer(M, name):
        for l in M.layers:
            if l.name == name:
                return l
        raise Exception(f"Layer: {name} not found in model.")

    @staticmethod
    def create_S_constant(model_params, station_i=0):
        S_inputs = tf.keras.Input(
            shape=(1,),
            batch_size=model_params['S']['n_batches'],
            name="S.input"
        )
        S = ConstantSTALayer(
            model_params['S']['n_batches'],
            model_params['S']['window_size'],
            model_params['S']['num_stations'],
            station_i=station_i,
            name="S.softmax"
        )(S_inputs)
        return S_inputs, S

    @staticmethod
    def create_S_random(model_params, seed=None, tiled=False, random_station_percentages=None):
        S_inputs = tf.keras.Input(
            shape=(1,),
            batch_size=model_params['S']['n_batches'],
            name="S.input"
        )
        S = RandomSTALayer(
            model_params['S']['n_batches'],
            model_params['S']['window_size'],
            model_params['S']['num_stations'],
            seed=seed,
            tiled=tiled,
            random_station_percentages=random_station_percentages,
            name="S.softmax"
        )(S_inputs)
        return S_inputs, S

    @staticmethod
    def create_C(C_inputs, model_params, return_model=False):
        C = layers.Dense(
            model_params['C']['hidden_size'],
            name="C.input",
            activation=model_params['C']['activation'],
            kernel_regularizer=regularizers.l2(0.01) if model_params['C']['use_kernel_regularization'] else None,
            activity_regularizer=regularizers.l1(0.01) if model_params['C']['use_activity_regularization'] else None
        )(C_inputs)
        C = layers.Flatten(name='C.flatten')(C)

        for i in range(model_params['C']['num_hidden_dense_layers']):
            C = layers.Dropout(model_params['C']['dropout'], name=f'C.dropout.{i}')(C)
            C = layers.Dense(
                model_params['C']['hidden_size'],
                name=f"C.hidden.{i}",
                activation=model_params['C']['activation'],
                kernel_regularizer=regularizers.l2(0.01) if model_params['C']['use_kernel_regularization'] else None,
                activity_regularizer=regularizers.l1(0.01) if model_params['C']['use_activity_regularization'] else None
            )(C)

        if model_params['C']['use_final_dropout']:
            C = layers.Dropout(model_params['C']['dropout'], name='C.dropout.final')(C)

        C = layers.Dense(
            model_params['C']['num_classes'],
            name="C.output",
        )(C)
        C = layers.Activation('softmax', name='C.softmax')(C)

        if return_model:
            return tf.keras.Model(inputs=[C_inputs], outputs=C)

        return C

    @staticmethod
    def create_SC(model_params, sub_model_to_train='S'):
        assert (sub_model_to_train in ['S', 'C'])

        csi_inputs = tf.keras.Input(
            shape=(
                model_params['S']['num_stations'],
                model_params['C']['num_subcarriers'],
                model_params['C']['window_size'],
            ),
            batch_size=model_params['S']['n_batches'],
            name='csi.input',
        )

        constant_station_i = model_params['S'].get('constant_station_i', None)
        if isinstance(constant_station_i, int):
            S_inputs, S = SCModel.create_S_constant(model_params, station_i=constant_station_i)
        elif model_params['S'].get('use_random_S', False):
            S_inputs, S = SCModel.create_S_random(
                model_params,
                model_params['S'].get('use_seeded_random_S', False),
                tiled=model_params['S'].get('use_tiled_random_S', False),
                random_station_percentages=model_params['S'].get('random_station_percentages', [0.2, 0.2, 0.2, 0.2, 0.2]),
            )
        else:
            S_inputs, S = SCModel.create_S(model_params)

        C_inputs = StationSelectionAggregator(
            n_batches=model_params['S']['n_batches'],
            structured=model_params['S'].get('structured', False),
        )(S, csi_inputs)
        C = SCModel.create_C(C_inputs, model_params)

        return tf.keras.Model(inputs=[S_inputs, csi_inputs], outputs=C)

    @staticmethod
    def train_SC(SC, dataset_params, experiment_params, model_params, train_sequence, test_sequence, n_epochs, _round, _run):
        # Fit model
        SC.fit(
            train_sequence,
            validation_data=test_sequence,
            epochs=n_epochs,
            verbose=1,
            shuffle=True,
            callbacks=[
                # LambdaCallback(on_epoch_end=SCModel.S_log_on_epoch_end(SC, train_sequence, _run)),  # TODO: add back
                LambdaCallback(on_epoch_end=SCModel.C_log_on_epoch_end(SC, train_sequence, _run)),
            ],
            batch_size=model_params['S']['n_batches'],
        )

        batch_r_hat = np.random.random([2, 1])

        return SC

    @staticmethod
    def train_C(SC, dataset_params, experiment_params, model_params, train_sequence, test_sequence, epochs, _round, _run):
        # Make a copy of SC where C is frozen and the loss is optimizing for S
        SC_hat = SCModel.create_SC(model_params, sub_model_to_train='C')

        # Copy parameters from SC to SC_hat
        SC_hat = SCModel.copyA2B(SC, SC_hat, set_C_trainable=True, set_S_trainable=False)

        # Compile model and set optimization
        SC_hat.compile(
            optimizer=model_params['C']['optimizer'],
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        print("losses", SC_hat.losses)
        print("metrics", SC_hat.metrics)

        print(SC_hat.summary(show_trainable=True))

        # Train SC
        SC_hat = SCModel.train_SC(SC, dataset_params, experiment_params, model_params, train_sequence, test_sequence, epochs, _round, _run)

        # Copy parameters from SC_hat to SC
        SC = SCModel.copyA2B(SC_hat, SC)

        return SC

    @staticmethod
    def C_log_on_epoch_end(SC, train_sequence, _run):
        def run(epoch, log):
            print("log", log)
            for k in log.keys():
                _k = k
                if 'val_' in k:
                    _k = f"validation {_k.replace('val_', '')}"
                _run.log_scalar(f"C.{_k}", log[k])

        return run

    @staticmethod
    def copyA2B(A, B, set_C_trainable=True, set_S_trainable=True):
        for i, l_B in enumerate(B.layers):
            # Copy Weights
            for l_A in A.layers:
                if l_A.name == l_B.name:
                    B.layers[i].set_weights(l_A.get_weights())

            # Set Trainable Layers
            name = B.layers[i].name
            B.layers[i].trainable = (set_C_trainable and 'C' in name) or (set_S_trainable and 'S' in name)
        return B

    @staticmethod
    def get_S_output(SC, r_batch):
        next_input = r_batch
        for l in SC.layers:
            next_input = l(next_input)
            if isinstance(next_input, list):
                next_input = next_input[0]
            if l.name == 'S.softmax':
                break
        return next_input.numpy()

    @staticmethod
    def evaluate_C(SC, train_sequence, test_sequence):
        return {
            'train': SC.evaluate(
                train_sequence,
                verbose=1,
                batch_size=1,
                return_dict=True,
            ),
            'test': SC.evaluate(
                test_sequence,
                verbose=1,
                batch_size=1,
                return_dict=True,
            ),
        }
