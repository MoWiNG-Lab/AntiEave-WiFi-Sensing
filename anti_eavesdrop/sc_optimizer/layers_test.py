import unittest
import numpy as np
import tensorflow as tf

from anti_eavesdrop.sc_optimizer.layers import StationSelectionAggregator


class TestStationSelectionAggregator(unittest.TestCase):
    def test_get_selection_mask(self):
        with tf.compat.v1.Session() as sess:
            num_batches = 2
            num_stations = 5
            num_subcarriers = 7
            num_timestamps = 3

            schedules = tf.constant(np.transpose([[  # Input: [batch,timestamp,station]
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]], [0, 2, 1]), dtype=tf.float32)
            schedules = tf.tile(schedules, [2, 1, 1])

            expected = np.zeros([num_batches, num_stations, num_subcarriers, num_timestamps])
            expected[:, 1, :, 0] = 1
            expected[:, 3, :, 1] = 1
            expected[:, 2, :, 2] = 1

            actual = sess.run(StationSelectionAggregator(n_batches=num_batches).get_selection_mask(
                    schedules, num_stations, num_subcarriers, num_timestamps
                ))

            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual((actual - expected).sum(), 0)



    def test_get_selected_csi(self):
        with tf.compat.v1.Session() as sess:
            window_size = 4
            n_subcarriers = 64
            schedules = tf.constant(np.transpose([  # Input: [batch,timestamp,station]
                [
                    [0, 0, 0, 0],  # Station-0
                    [0, 0, 0, 0],  # Station-1
                    [0, 1, 0, 0],  # Station-2
                    [1, 0, 1, 0],  # Station-3
                    [0, 0, 0, 1],  # Station-4
                ],
                [
                    [0, 0, 0, 1],  # Station-0
                    [0, 0, 1, 0],  # Station-1
                    [0, 1, 0, 0],  # Station-2
                    [0, 0, 0, 0],  # Station-3
                    [1, 0, 0, 0],  # Station-4
                ],
            ], [0, 2, 1]), dtype=tf.float32)

            csi = tf.constant(np.array([[
                np.ones([n_subcarriers, window_size]) * 0.09,  # CSI from Station-0
                np.ones([n_subcarriers, window_size]) * 1.11,  # CSI from Station-1
                np.ones([n_subcarriers, window_size]) * 2.22,  # CSI from Station-2
                np.ones([n_subcarriers, window_size]) * 3.33,  # CSI from Station-3
                np.ones([n_subcarriers, window_size]) * 4.44,  # CSI from Station-4
            ]]), dtype=tf.float32)
            csi = tf.tile(csi, [2, 1, 1, 1])

            layer = StationSelectionAggregator(n_batches=2)
            output = sess.run(layer.call(schedules, csi))
            csi = sess.run(csi)
            print(output)
            print("output.shape", output.shape)

            # Tensor shapes
            # output[batch_i, time_t, subcarrier_s]
            # csi[batch_i, station_i, time_t, subcarrier_s]
            batch_count = 2
            self.assertEquals(output.shape, (batch_count, window_size, n_subcarriers))

            # Should return CSI from stations in this order:
            # batch-0: [3, 2, 3, 4]
            self.assertEquals(output[0][0].tolist(), csi[0,3,:,0].tolist())
            self.assertEquals(output[0][1].tolist(), csi[0,2,:,0].tolist())
            self.assertEquals(output[0][2].tolist(), csi[0,3,:,0].tolist())
            self.assertEquals(output[0][3].tolist(), csi[0,4,:,0].tolist())

            # batch-1: [4, 2, 1, 0]
            self.assertEquals(output[1][0].tolist(), csi[0,4,:,0].tolist())
            self.assertEquals(output[1][1].tolist(), csi[0,2,:,0].tolist())
            self.assertEquals(output[1][2].tolist(), csi[0,1,:,0].tolist())
            self.assertEquals(output[1][3].tolist(), csi[0,0,:,0].tolist())



    def test_get_selected_csi__structured(self):
        # If the RX knows which TX is transmitting, then the input data can be structured in such a way.
        with tf.compat.v1.Session() as sess:
            window_size = 4
            n_subcarriers = 64
            num_stations = 5
            schedules = tf.constant(np.transpose([  # Input: [batch,timestamp,station]
                [
                    [0, 0, 0, 0],  # Station-0
                    [0, 0, 0, 0],  # Station-1
                    [0, 1, 0, 0],  # Station-2
                    [1, 0, 1, 0],  # Station-3
                    [0, 0, 0, 1],  # Station-4
                ],
                [
                    [0, 0, 0, 1],  # Station-0
                    [0, 0, 1, 0],  # Station-1
                    [0, 1, 0, 0],  # Station-2
                    [0, 0, 0, 0],  # Station-3
                    [1, 0, 0, 0],  # Station-4
                ],
            ], [0, 2, 1]), dtype=tf.float32)

            csi = tf.constant(np.array([[
                np.ones([n_subcarriers, window_size]) * 0.09,  # CSI from Station-0
                np.ones([n_subcarriers, window_size]) * 1.11,  # CSI from Station-1
                np.ones([n_subcarriers, window_size]) * 2.22,  # CSI from Station-2
                np.ones([n_subcarriers, window_size]) * 3.33,  # CSI from Station-3
                np.ones([n_subcarriers, window_size]) * 4.44,  # CSI from Station-4
            ]]), dtype=tf.float32)
            csi = tf.tile(csi, [2, 1, 1, 1])

            layer = StationSelectionAggregator(n_batches=2, structured=True)
            output = sess.run(layer.call(schedules, csi))
            csi = sess.run(csi)
            print(output)
            print("output.shape", output.shape)

            # Tensor shapes
            # output[batch_i, time_t, subcarrier_s]
            # csi[batch_i, station_i, time_t, subcarrier_s]
            batch_count = 2
            self.assertEquals(output.shape, (batch_count, num_stations, window_size, n_subcarriers))

            # Should return CSI from stations in this order:
            # batch-0: [3, 2, 3, 4]
            self.assertEquals(output[0][0][0].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][1][0].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][2][0].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][3][0].tolist(), csi[0,3,:,0].tolist())
            self.assertEquals(output[0][4][0].tolist(), np.zeros(n_subcarriers).tolist())

            self.assertEquals(output[0][0][1].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][1][1].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][2][1].tolist(), csi[0,2,:,0].tolist())
            self.assertEquals(output[0][3][1].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][4][1].tolist(), np.zeros(n_subcarriers).tolist())

            self.assertEquals(output[0][0][2].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][1][2].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][2][2].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][3][2].tolist(), csi[0,3,:,0].tolist())
            self.assertEquals(output[0][4][2].tolist(), np.zeros(n_subcarriers).tolist())

            self.assertEquals(output[0][0][3].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][1][3].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][2][3].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][3][3].tolist(), np.zeros(n_subcarriers).tolist())
            self.assertEquals(output[0][4][3].tolist(), csi[0,4,:,0].tolist())
