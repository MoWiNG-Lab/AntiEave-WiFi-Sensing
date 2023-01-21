from tensorflow import keras
import math
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class Batch3DTo4DSequence(keras.utils.Sequence):
    """
    Specialized replacement of Batch2DTo3DSequence.

    This generator takes in the 3D CSI data (Shape: [transmitter, total-time-steps, subcarriers]) to generate batches of
    4D CSI data (Shape: [transmitter, total-time-step, window-size, subcarrier]).
    """

    def __init__(self, X_all, y, meta, y_columns, all_y_columns, batch_r, num_timesteps, batch_size=1000, should_shuffle=True, use_training=False, use_testing=False, name=None, use_regression=False):
        self.X_all = X_all
        self.y = y
        self.meta = meta
        self.y_columns = y_columns
        self.all_y_columns = all_y_columns
        self.batch_r = batch_r
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        self.name = name
        self.use_regression = use_regression

        self.y_column_indices = [self.all_y_columns.index(c) for c in self.y_columns]

        self.all_indices = np.concatenate([self.y[self.y[c] == 1].index for c in self.y_columns])
        self.all_indices.sort()

        training_indices = list(self.y[self.meta['is_training_sample'] == 1].index)
        testing_indices = list(self.y[self.meta['is_training_sample'] == 0].index)

        self.all_indices = [i for i in self.all_indices if i < len(self.y)-self.num_timesteps]

        indices_to_remove = []
        if not use_training:
            indices_to_remove += training_indices
        if not use_testing:
            indices_to_remove += testing_indices

        self.all_indices = list(set(self.all_indices) - set(indices_to_remove))

        self.should_shuffle = should_shuffle
        if self.should_shuffle:
            self.all_indices = shuffle(self.all_indices)

    def __len__(self):
        if self.batch_size:
            return math.floor(len(self.all_indices) / self.batch_size)  # We lose some samples this way...
            # return math.ceil(len(self.all_indices) / self.batch_size)
        else:
            return 1

    def __getitem__(self, idx):
        if self.batch_size:
            current_batch_indices = self.all_indices[(idx*self.batch_size):((idx+1)*self.batch_size)]
        else:
            current_batch_indices = self.all_indices

        # Batch for input into S
        # batch_r = np.random.random([self.batch_size, 1])  # TODO: Put back.
        batch_r = self.batch_r
        # Batch for input into C and output for SC
        batch_X, batch_y = self.create_4d(current_batch_indices)

        return [batch_r, batch_X], batch_y

    def create_4d(self, batch_indices):
        # Shape: [num_batches, num_stations, num_timesteps, num_subcarriers]
        def create_timeframe(i):
            return self.X_all[:, i:i + self.num_timesteps, :]

        batch_X = pd.Series(batch_indices).apply(create_timeframe)
        batch_y = self.y.iloc[batch_indices, self.y_column_indices]

        #
        # Convert to numpy and swap CSI axes
        #
        batch_X = np.array(batch_X.to_numpy().tolist(), dtype=float)
        batch_y = np.array(batch_y.to_numpy().tolist(), dtype=np.int32)
        batch_X = np.swapaxes(batch_X, 2, 3)

        return batch_X, batch_y

    def on_epoch_end(self):
        if self.should_shuffle:
            self.all_indices = shuffle(self.all_indices)
