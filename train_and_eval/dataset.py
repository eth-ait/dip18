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

This class provides a basic interface to feed samples by using tensorflow's input pipeline (i.e., queues), hiding data
I/O latency bu using threads.

A `Dataset` object is given to `DataFeederTF` object which runs `sample_generator` method to enqueue the data queue.
The `sample_generator` method returns a generator yielding one sample at a time with shape and type specified by
`sample_shape` and `sample_tf_type`.

The way the data is passed is not restricted. A child class can read the data from numpy array, list, dictionary, etc.
"""
import numpy as np
import tensorflow as tf

from constants import Constants
from data_operators import Operator

C = Constants()


class BaseDataset(object):
    """
    Acts as a data container. Loads and parses data, and provides basic functionality.
    """
    def __init__(self, data_path):
        if isinstance(data_path, str):
            self.data_dict = dict(np.load(data_path))
        elif isinstance(data_path, dict):
            self.data_dict = data_path
        else:
            raise Exception("Data type isn't recognized.")

        self.num_samples = None
        self.sample_shape = None
        self.sample_np_type = None
        self.sample_tf_type = None
        self.sample_key = None

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): that yields one sample consisting of a list of data elements.
        """
        raise NotImplementedError('Method is abstract.')

    def batch_generator(self, batch_size, shuffle=True, drop_last_batch=True):
        """
        Creates a generator object which returns a batch of samples at a time.

        Returns:
            (generator): that yields a batch of samples.
        """
        raise NotImplementedError('Method is abstract.')


class ImuDataset(BaseDataset):
    def __init__(self, data_path, var_len_seq=False, preprocessing_ops=None, data_stats=None):
        super(ImuDataset, self).__init__(data_path)

        preprocessing_ops = preprocessing_ops or {}

        self.samples_smpl = self.data_dict['smpl_pose']
        self.samples_acc = self.data_dict['acceleration']
        self.samples_ori = self.data_dict['orientation']

        self.file_id = self.data_dict.get('file_id', None)
        self.data_id = self.data_dict.get('data_id', None)

        self.size_smpl = self.samples_smpl[0].shape[1]
        self.size_acc = self.samples_acc[0].shape[1]
        self.size_ori = self.samples_ori[0].shape[1]

        self.num_samples = len(self.samples_ori)
        self.input_feature_size = 0
        self.target_feature_size = 0
        self.sequence_lengths = self.__extract_seq_len()

        input_keys = ['orientation', 'acceleration']
        target_keys = ['smpl_pose']

        if "orientation" in input_keys:
            self.input_feature_size += self.size_ori
        if "acceleration" in input_keys:
            self.input_feature_size += self.size_acc
        if "smpl_pose" in input_keys:
            self.input_feature_size += self.size_smpl

        if "orientation" in target_keys:
            self.target_feature_size += self.size_ori
        if "acceleration" in target_keys:
            self.target_feature_size += self.size_acc
        if "smpl_pose" in target_keys:
            self.target_feature_size += self.size_smpl

        if data_stats is None:
            # take the stats stored in the data set
            data_stats = self.data_dict.get('statistics').tolist() if 'statistics' in self.data_dict else {}

        self.ori_stats = data_stats['orientation']
        self.acc_stats = data_stats['acceleration']
        self.smpl_stats = data_stats['smpl_pose']

        preprocessing_ops[C.PP_ZERO_MEAN_NORM] = True if preprocessing_ops[C.PP_IMU_ORI_NORM] else False
        self.preprocessor_ori = Operator.create(**{**preprocessing_ops, **self.ori_stats})
        preprocessing_ops[C.PP_ZERO_MEAN_NORM] = True if preprocessing_ops[C.PP_IMU_ACC_NORM] else False
        self.preprocessor_acc = Operator.create(**{**preprocessing_ops, **self.acc_stats})
        preprocessing_ops[C.PP_ZERO_MEAN_NORM] = True if preprocessing_ops[C.PP_IMU_SMPL_NORM] else False
        self.preprocessor_smpl = Operator.create(**{**preprocessing_ops, **self.smpl_stats})

        # Models require input and target dimensionality. `*_dims` members are useful if the inputs and targets are
        # concatenation of different modalities. They are used to split the input/target into components by the model.
        self.input_dims = [self.input_feature_size]
        self.target_dims = [self.target_feature_size]

        # The dimensions with None will be padded if seq_len isn't passed.
        self.sequence_length = None if var_len_seq else self.__get_seq_len()
        self.is_dynamic = self.sequence_length is None

        # sequence length, [orientation, acceleration], smpl_pose
        self.sample_shape = [[], [self.sequence_length, self.size_ori+self.size_acc], [self.sequence_length, self.size_smpl]]
        self.sample_np_type = [np.int32, np.float32, np.float32]
        self.sample_key = [C.PL_SEQ_LEN, C.PL_INPUT, C.PL_TARGET]

    def unnormalize(self, sample):
        """
        Doesn't do anything for the moment.
        """
        raise NotImplementedError()

    def undo_preprocess_smpl(self, smpl_samples):
        if self.preprocessor_smpl is not None:
            if not isinstance(smpl_samples, list):
                smpl_samples = [smpl_samples]

            outputs = []
            for smpl in smpl_samples:
                smpl, _ = self.preprocessor_smpl.undo(np.expand_dims(smpl, axis=0))
                outputs.append(smpl[0])

            return outputs
        else:
            return smpl_samples

    def prepare_for_visualization(self, sample):
        """
        Prepare the given sample for visualization by undoing normalization and representation related operations.
        """
        return self.unnormalize(sample)

    def sample_generator(self):
        """
        Creates a generator object which returns one data sample at a time. It is used by DataFeeder objects.

        Returns:
            (generator): each sample is a list of data elements.
        """
        for seq_len, ori, acc, smpl in zip(self.sequence_lengths, self.samples_ori, self.samples_acc, self.samples_smpl):

            if self.preprocessor_ori is not None:
                ori, _ = self.preprocessor_ori.apply(np.expand_dims(ori, axis=0))
                ori = ori[0]

            if self.preprocessor_acc is not None:
                acc, _ = self.preprocessor_acc.apply(np.expand_dims(acc, axis=0))
                acc = acc[0]

            if self.preprocessor_smpl is not None:
                smpl, _ = self.preprocessor_smpl.apply(np.expand_dims(smpl, axis=0))
                smpl = smpl[0]

            inputs = np.concatenate([ori, acc], axis=-1)
            targets = smpl

            yield [seq_len, inputs, targets]

    def fetch_sample(self, sample_idx):
        """
        Prepares one data sample (i.e. return of sample_generator) given index.

        Args:
            sample_idx:

        Returns: the sample
        """
        smpl = self.samples_smpl[sample_idx]
        acc = self.samples_acc[sample_idx]
        ori = self.samples_ori[sample_idx]
        seq_len = self.sequence_lengths[sample_idx]

        if self.preprocessor_ori is not None:
            ori, _ = self.preprocessor_ori.apply(np.expand_dims(ori, axis=0))
            ori = ori[0]

        if self.preprocessor_acc is not None:
            acc, _ = self.preprocessor_acc.apply(np.expand_dims(acc, axis=0))
            acc = acc[0]

        if self.preprocessor_smpl is not None:
            smpl, _ = self.preprocessor_smpl.apply(np.expand_dims(smpl, axis=0))
            smpl = smpl[0]

        return [seq_len, np.concatenate([ori, acc], axis=-1), smpl]

    def batch_generator(self, batch_size, epoch=1, shuffle=True, drop_last_batch=True):
        """
        Creates a generator object which returns a batch of samples at a time.

        Args:
            batch_size (int): how many samples per batch to load.
            epoch (int): number of iterations over all data samples.
            shuffle (bool): set to True to have the data reshuffled at every epoch (default: True).
            drop_last_batch (bool): set to True to drop the last incomplete batch, if the dataset size is not divisible
                by the batch size (default: True).
        Returns:
            (generator):
        """
        for e in range(epoch):
            if shuffle:
                indices = np.random.permutation(self.num_samples)
            else:
                indices = np.arange(self.num_samples)

            num_samples = len(indices)
            if drop_last_batch:
                num_samples -= num_samples%batch_size

            for i in range(0, num_samples, batch_size):
                batch_sample_idx = indices[i:i + batch_size]
                batch_seq_len = self.sequence_lengths[batch_sample_idx]
                max_len = batch_seq_len.max()

                batch_acc = np.zeros((batch_size, max_len, self.size_acc))
                batch_ori = np.zeros((batch_size, max_len, self.size_ori))
                batch_smpl = np.zeros((batch_size, max_len, self.size_smpl))
                batch_mask = np.zeros((batch_size, max_len))
                for id, sample_idx in enumerate(batch_sample_idx):
                    batch_acc[id] = self.samples_acc[sample_idx]
                    batch_smpl[id] = self.samples_ori[sample_idx]
                    batch_ori[id] = self.samples_smpl[sample_idx]
                    batch_mask[id] = np.ones((batch_seq_len[id]))

                yield [batch_seq_len, np.concatenate([batch_ori, batch_acc], axis=-1), batch_smpl, batch_mask]

    def __extract_seq_len(self):
        """
        Returns (np.array):
            List of lengths of each sequence sample in the dataset.
        """
        return np.array([s.shape[0] for s in self.samples_ori], dtype=np.int32)

    def __get_seq_len(self):
        """
        Returns (int or None):
            Sequence length of samples in the dataset. If the samples are variable-length then returns None. If dataset
            is already padded (i.e., preprocessing) then returns the fixed sample length, because padding is not
            required.
        """
        if max(self.sequence_lengths) == min(self.sequence_lengths):
            return min(self.sequence_lengths)
        else:
            return None


class ImuDatasetTF(ImuDataset):
    """
    To decouple tensorflow routines from standard python routines so that dataset class can still be used with
    other frameworks.
    """
    def __init__(self, data_path, var_len_seq=False, preprocessing_ops=None, data_stats=None):
        super(ImuDatasetTF, self).__init__(data_path,
                                           var_len_seq=var_len_seq,
                                           preprocessing_ops=preprocessing_ops,
                                           data_stats=data_stats)

        # Add tensorflow data types.
        self.sample_tf_type = [tf.int32, tf.float32, tf.float32]
