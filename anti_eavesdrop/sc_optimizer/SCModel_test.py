import unittest
import numpy as np
import tensorflow as tf

from anti_eavesdrop.sc_optimizer.SCModel import SCModel


class TestSCModel(unittest.TestCase):

    def test_createS(self):
        S__num_stations = 5
        S__window_size = 10
        C__num_subcarriers = 64
        model_params = {
            'S': {
                'num_stations': S__num_stations,
                'window_size': S__window_size,
                'n_batches': 1,
                'num_hidden_dense_layers': 2,
                'hidden_size': 50,
                'regularization_losses': [],
            },
        }

        S_inputs, S = SCModel.create_S(model_params)
        S = tf.keras.Model(inputs=S_inputs, outputs=S)
        silly_loss = lambda a, b: 0.0
        loss = silly_loss
        S.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

        y_hat = S.predict([
            np.array([[1.0]]),
            # np.random.random([1, S__num_stations, S__window_size, C__num_subcarriers])
        ])

        self.assertEquals(y_hat.shape, (1, S__window_size, S__num_stations))

    def test_createSC(self):
        S__num_stations = 5
        S__window_size = 10
        C__num_subcarriers = 64
        C__num_classes = 5
        model_params = {
            'S': {
                'num_stations': S__num_stations,
                'window_size': S__window_size,
                'n_batches': 1,
                'num_hidden_dense_layers': 2,
                'hidden_size': 50,
                'regularization_losses': [],
            },
            'C': {
                'num_subcarriers': 64,
                'window_size': S__window_size,
                'hidden_size': 100,
                'num_hidden_dense_layers': 2,
                'use_final_dropout': True,
                'num_classes': C__num_classes,
                'learning_rate': 1e-3,
                'dropout': 0.4,
                'use_kernel_regularization': True,
                'use_activity_regularization': True,
                'optimizer': 'adam',
                'activation': 'relu',
            },
        }

        SC = SCModel.create_SC(model_params)

        y_hat = SC.predict([
            np.array([[1.0]]),
            np.random.random([1, S__num_stations, C__num_subcarriers, S__window_size])
        ])

        self.assertEqual(y_hat.shape, (1, C__num_classes))
