"""
DIP: training, evaluating and running of deep inertial poser.
Copyright (C) 2018 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging

import threading
from dataset import BaseDataset
from constants import Constants
C = Constants()


class DataFeederTF(object):
    """
    Creates a tensorflow feeder in computational graph. The output variables are defined by the input dataset object.
    Uses threads to enqueue data asynchronously, and hides I/O latency.
    """

    def __init__(self, dataset, num_epochs, batch_size=16, queue_capacity=512, shuffle=True,
                 allow_smaller_final_batch=False):
        assert(isinstance(dataset, BaseDataset))

        self.dataset = dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.queue_capacity = queue_capacity
        self.epoch = 0
        self.allow_smaller_final_batch = allow_smaller_final_batch

        self.queue_placeholders_dict = {}
        self.queue_placeholders = [] # One-to-one correspondence with dataset.sample_* members.
        self.num_data_variables = len(self.dataset.sample_shape)

        for i in range(self.num_data_variables):
            self.queue_placeholders.append(tf.placeholder(self.dataset.sample_tf_type[i], shape=self.dataset.sample_shape[i]))
            self.queue_placeholders_dict[self.dataset.sample_key[i]] = self.queue_placeholders[-1]

        # Tensorflow Queues complain if we don't have fully defined tensors. In other words, we need to have static
        # shapes. However, batch generators such as tf.train.batch need to know tensor rank. Otherwise, it makes random
        # placeholder assignments (i.e., input is mapped to seq_len placeholder). This seems to be a Tensorflow bug.
        if shuffle:
            self.input_queue = tf.RandomShuffleQueue(queue_capacity,
                                                     min_after_dequeue=int(queue_capacity/2),
                                                     dtypes=self.dataset.sample_tf_type,
                                                     names=self.dataset.sample_key)
        else:
            self.input_queue = tf.FIFOQueue(queue_capacity,
                                            dtypes=self.dataset.sample_tf_type,
                                            names=self.dataset.sample_key)

        self.enqueue_op = self.input_queue.enqueue(self.queue_placeholders_dict)
        self.dequeue_op = self.input_queue.dequeue()

        # Set tensor shapes here.
        for i in range(self.num_data_variables):
            self.dequeue_op[self.dataset.sample_key[i]].set_shape(self.dataset.sample_shape[i])

    def batch_queue(self, dynamic_pad=True, queue_capacity=512, queue_threads=4, name="batch_generator"):
        """
        A plain feeder is used and range of sequence lengths in a batch will be arbitrary.

        Args:
            dynamic_pad:
            queue_capacity:
            queue_threads:

        Returns:

        """
        self.batch = tf.train.batch(self.dequeue_op,
                                    batch_size=self.batch_size,
                                    capacity=int(queue_capacity / 2) + (queue_threads + 2) * self.batch_size,
                                    num_threads=queue_threads,
                                    enqueue_many=False,
                                    dynamic_pad=dynamic_pad,
                                    allow_smaller_final_batch=self.allow_smaller_final_batch,
                                    name=name)
        return self.batch

    def batch_queue_bucket(self, buckets, dynamic_pad=True, queue_capacity=128, queue_threads=4,
                           name="batch_generator_bucket"):
        """
        Samples are first bucketed with respect to the sequence length. In this case the first entry of each sample in
        the dataset must be the sequence length.

        Args:
            buckets (list): a list of bucket boundaries (i.e., the edges of the buckets to use when bucketing samples)
            dynamic_pad:
            queue_capacity:
            queue_threads:

        Returns:
        """
        batch_seq_lens, self.batch = tf.contrib.training.bucket_by_sequence_length(
                                    input_length=self.dequeue_op[C.PL_SEQ_LEN],
                                    tensors=self.dequeue_op,
                                    batch_size=self.batch_size,
                                    bucket_boundaries=buckets,
                                    num_threads=queue_threads,
                                    capacity=queue_capacity,
                                    bucket_capacities=[self.batch_size*3]*(len(buckets)+1),
                                    dynamic_pad=dynamic_pad,
                                    allow_smaller_final_batch=False,
                                    name=name)
        return self.batch

    def __enqueue(self, tf_session, tf_coord):
        sample_generator = self.dataset.sample_generator()
        while not tf_coord.should_stop():
            try:
                sample = next(sample_generator)
                feed_dict = {pl: val for pl, val in zip(self.queue_placeholders, sample)}
                tf_session.run(self.enqueue_op, feed_dict=feed_dict)
            except StopIteration:
                sample_generator = self.dataset.sample_generator()
            except tf.errors.CancelledError:
                pass

    def init(self, tf_session, tf_coord):
        self.enqueue_threads = threading.Thread(target=self.__enqueue, args=[tf_session, tf_coord])
        self.enqueue_threads.start()


class TFStagingArea(object):

    def __init__(self, tensors, device_name=None):
        if device_name is None:
            self._staging_area = self._create_staging_area(tensors)
        else:
            with tf.device(device_name):
                self._staging_area = self._create_staging_area(tensors)
        self._preload_op = self._staging_area.put(tensors)
        self._tensors = self._staging_area.get()

    def _create_staging_area(self, tensors):
        names, dtypes, shapes = [], [], []
        for name, tensor in tensors.items():
            dtypes.append(tensor.dtype)
            shapes.append(tensor.shape)
            names.append(name)

        return tf_staging.StagingArea(dtypes=dtypes, shapes=shapes, names=names)

    @property
    def preload_op(self):
        return self._preload_op

    @property
    def tensors(self):
        return self._tensors
