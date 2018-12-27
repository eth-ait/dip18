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
import numpy as np
from constants import Constants
C = Constants()


class Operator(object):
    """
    Carries out pre-processing operations.
    """

    def __init__(self, operator_obj=None):
        self.operator_obj = operator_obj
        self.side_effects = {}

    def apply(self, input_data, target_data=None):
        """
        Applies a preprocessing operation on given input and target samples (if not None).

        Args:
            input_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)
            target_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)

        Returns:
        """
        return input_data, target_data

    def undo(self, input_data, target_data=None):
        """
        Undo the preprocessing operation if it is stateless. Otherwise, implements identity function.

        Args:
            input_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)
            target_data (np.ndarray): shape of (batch_size, sequence_length, feature_size)

        Returns:
        """
        return input_data, target_data

    @staticmethod
    def create(**kwargs):
        operator_obj = Operator()

        if kwargs.get(C.PP_ZERO_MEAN_NORM, False):
            operator_obj = NormalizeZeroMeanUnitVariance(data_mean=kwargs['mean_channel'], data_std=kwargs['std_channel'], operator_obj=operator_obj)

        return operator_obj


class NormalizeZeroMeanUnitVariance(Operator):
    def __init__(self, data_mean, data_std, operator_obj=None):
        super(NormalizeZeroMeanUnitVariance, self).__init__(operator_obj)
        self.data_mean = data_mean
        self.data_std = data_std

        self.side_effects = operator_obj.side_effects

    def apply(self, input_data, target_data=None):
        input_operated, target_operated = self.operator_obj.apply(input_data, target_data)

        input_operated = (input_operated - self.data_mean) / self.data_std
        if target_operated is not None:
            target_operated = (target_operated - self.data_mean) / self.data_std

        return input_operated, target_operated

    def undo(self, input_data, target_data=None):
        input_reverted = input_data*self.data_std + self.data_mean
        if target_data is not None:
            target_reverted = target_data*self.data_std + self.data_mean
        else:
            target_reverted = target_data

        input_reverted, target_reverted = self.operator_obj.undo(input_reverted, target_reverted)

        return input_reverted, target_reverted
