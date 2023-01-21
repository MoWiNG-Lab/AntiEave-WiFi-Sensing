import numpy as np
import tensorflow as tf
from tensorflow import keras

class RandomSTALayer(keras.layers.Layer):
    """Layer that always outputs random stations."""

    def __init__(self, batch_size, window_size, num_stations, seed, tiled, random_station_percentages=None, **kwargs):
        super(RandomSTALayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_stations = num_stations
        self.seed = seed
        self.tiled = tiled
        if random_station_percentages is None:
            self.random_station_percentages = [0.2, 0.2, 0.2, 0.2, 0.2]
        else:
            self.random_station_percentages = random_station_percentages
            assert abs(1.0 - sum(self.random_station_percentages)) < 1e-5

    def call(self, inputs):
        out = np.zeros([
            self.batch_size,
            self.window_size,
            self.num_stations,
        ])

        if self.seed is not None:
            np.random.seed(9)

        def make_random_schedule(probabilities, window_size):
            R = np.random.random([1, window_size])
            return np.apply_along_axis(lambda x: sum(x[0] > np.cumsum(probabilities)), 0, R)

        if self.tiled:
            R = make_random_schedule(self.random_station_percentages, self.window_size)
            R = np.reshape(R, [1, R.shape[0]])
            R = np.tile(R, [self.batch_size, 1])
        else:
            R = []
            for _ in range(self.batch_size):
                R.append(make_random_schedule(self.random_station_percentages, self.window_size))
            R = np.array(R)

        for i in range(self.batch_size):
            for j in range(self.window_size):
                for k in range(self.num_stations):
                    if R[i, j] == k:
                        out[i, j, k] = 1

        return tf.constant(out, dtype=tf.float32)


class ConstantSTALayer(keras.layers.Layer):
    """Layer that always outputs one station."""

    def __init__(self, batch_size, window_size, num_stations, station_i, **kwargs):
        super(ConstantSTALayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_stations = num_stations
        self.station_i = station_i

    def call(self, inputs):
        out = np.zeros([
            self.batch_size,
            self.window_size,
            self.num_stations,
        ])
        out[:, :, self.station_i] = 1.0
        return tf.constant(out, dtype=tf.float32)


class StationSelectionAggregator(keras.layers.Layer):
    """Layer that aggregates CSI from $n$ stations based on the one selected station at every time step."""

    def __init__(self, n_batches, structured=False, **kwargs):
        super(StationSelectionAggregator, self).__init__(**kwargs)
        self.n_batches = n_batches
        self.structured = structured

    def get_selection_mask(self, schedules, num_stations, num_subcarriers, num_timestamps):
        x = schedules
        x = tf.transpose(x, [0, 2, 1])
        x = tf.reshape(x, [x.shape[0], x.shape[1], 1, x.shape[2]])
        return tf.tile(x, [1, 1, num_subcarriers, 1])

    def get_selected_csi(self, schedules, csi):
        n_batches, n_stations, n_subcarriers, window_size = csi.shape
        n_batches2, n_stations2, window_size2 = schedules.shape

        mask = self.get_selection_mask(schedules, n_stations, n_subcarriers, window_size)
        if self.structured:
            out = tf.einsum('btsw,btsw->btsw', csi, mask)
            out = tf.transpose(out, [0, 1, 3, 2])
            return out
        else:
            out = tf.einsum('btsw,btsw->bsw', csi, mask)
            out = tf.transpose(out, [0, 2, 1])
            return out

    def call(self, schedule, csi):
        return self.get_selected_csi(schedule, csi)
