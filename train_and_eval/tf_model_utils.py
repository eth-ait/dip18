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
import tensorflow as tf
from constants import Constants

C = Constants()


def get_activation_fn(type=C.RELU):
    """
    Return tensorflow activation function given string name.
    """
    if type == C.RELU:
        return tf.nn.relu
    elif type == C.ELU:
        return tf.nn.elu
    elif type == C.TANH:
        return tf.nn.tanh
    elif type == C.SIGMOID:
        return tf.nn.sigmoid
    elif type == C.SOFTPLUS:
        return tf.nn.softplus
    elif type == C.SOFTMAX:
        return tf.nn.softmax
    elif type == C.LRELU:
        return lambda x: tf.nn.leaky_relu(x, alpha=1./3.)
    elif type == C.CLRELU:
        with tf.name_scope('ClampedLeakyRelu'):
            return lambda x: tf.clip_by_value(tf.nn.leaky_relu(x, alpha=1./3.), -3.0, 3.0)
    elif type is None:
        return None
    else:
        raise Exception("Activation function is not implemented.")


def linear(input_, output_size, activation_fn=None, batch_norm=False, is_training=True):
    """
    Creates a linear layer.

    Args:
        input_:
        output_size:
        activation_fn : tensorflow activation function such as tf.nn.relu, tf.nn.sigmoid, etc.
        batch_norm (bool): whether use batch normalization layer or not.
        is_training (bool): whether in training mode or not.
    """
    dense_layer = tf.layers.dense(input_, output_size)
    if batch_norm is True and activation_fn is not None:
        dense_layer = tf.layers.batch_normalization(dense_layer, axis=1, training=is_training)

    if isinstance(activation_fn, str):
        activation_fn = get_activation_fn(activation_fn)
    if activation_fn is not None:
        dense_layer = activation_fn(dense_layer)
    return dense_layer


def fully_connected_layer(input_, is_training=True, **kwargs):
    """
    Creates fully connected layers.

    Args:
        input_:
        is_training (bool): whether in training mode or not.
        **kwargs: `size`, `activation_fn`, `num_layers`

    Returns:
    """
    activation_fn = get_activation_fn(kwargs.get('activation_fn', C.RELU))
    num_layers = kwargs.get('num_layers', 1)
    hidden_size = kwargs.get('size', 256)
    use_batch_norm = kwargs.get('use_batch_norm', False)

    hidden_layer = input_
    for i in range(num_layers):
        hidden_layer = linear(hidden_layer, hidden_size, activation_fn=activation_fn, batch_norm=use_batch_norm, is_training=is_training)
    return hidden_layer


def get_reduce_loss_func(type="sum_mean", seq_len=None):
    """
    Return the reduce function for the loss computation
    Args:
        type: "sum_mean", "mean", "sum".
        seq_len: sequence length
    """
    def reduce_sum_mean(loss):
        """
        Average batch loss. First calculates per sample loss by summing over the second and third dimensions and then
        takes the average.
        """
        rank = len(loss.get_shape())
        if rank > 3 or rank < 2:
            raise Exception("Loss rank must be 2 or 3.")

        if rank == 3:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]))
        elif rank == 2:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=[1]))

    def reduce_mean_per_step(loss):
        """
        First calculates average loss per sample (loss per step), and then takes average over samples. Loss per step
        requires sequence length. If all samples have the same sequence length then this is equivalent to `mean`.
        """
        rank = len(loss.get_shape())
        if rank > 3 or rank < 2:
            raise Exception("Loss rank must be 2 or 3.")

        # Calculate loss per step.
        if rank == 3:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1, 2])/tf.cast(seq_len, tf.float32)
        elif rank == 2:
            step_loss_per_sample = tf.reduce_sum(loss, axis=[1])/tf.cast(seq_len, tf.float32)
        # Calculate average (per step) sample loss.
        return tf.reduce_mean(step_loss_per_sample)

    if type == C.R_MEAN_SEQUENCE:
        return reduce_sum_mean
    elif type == C.R_SUM:
        return tf.reduce_sum
    elif type == C.R_MEAN:
        return tf.reduce_mean
    elif type == C.R_MEAN_STEP:
        return reduce_mean_per_step


def get_rnn_cell(**kwargs):
    """
    Creates an rnn cell object.

    Args:
        **kwargs: must contain `cell_type`, `size` and `num_layers` key-value pairs. `dropout_keep_prob` is optional.
            `dropout_keep_prob` can be a list of ratios where each cell has different dropout ratio in a stacked
            architecture. If it is a scalar value, then the whole architecture (either a single cell or stacked cell)
            has one DropoutWrapper.

    Returns:
    """

    cell_type = kwargs['cell_type']
    size = kwargs['size']
    num_layers = kwargs['num_layers']
    dropout_keep_prob = kwargs.get('dropout_keep_prob', 1.0)

    separate_dropout = False
    if isinstance(dropout_keep_prob, list) and len(dropout_keep_prob) == num_layers:
        separate_dropout = True

    if cell_type == C.LSTM:
        rnn_cell_constructor = tf.nn.rnn_cell.LSTMCell
    elif cell_type.lower() == C.GRU:
        rnn_cell_constructor = tf.nn.rnn_cell.GRUCell
    elif cell_type.lower() == C.LayerNormLSTM:
        rnn_cell_constructor = tf.contrib.rnn.LayerNormBasicLSTMCell
    else:
        raise Exception("Unsupported RNN Cell.")

    rnn_cells = []
    for i in range(num_layers):
        cell = rnn_cell_constructor(size)
        if separate_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_keep_prob[i],
                                                 output_keep_prob=dropout_keep_prob,
                                                 state_keep_prob=1,
                                                 dtype=tf.float32,
                                                 seed=1)
        rnn_cells.append(cell)

    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
    else:
        cell = rnn_cells[0]

    if separate_dropout and dropout_keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob,
                                             state_keep_prob=1,
                                             dtype=tf.float32,
                                             seed=1)
    return cell


def logli_normal_isotropic(x, mu, sigma):
    """
    Isotropic Gaussian log-likelihood loss.
    """
    with tf.name_scope('logli_normal_isotropic'):
        var = tf.maximum(1e-6, tf.square(sigma))
        result = -0.5 * (tf.log(2*np.pi*var) + tf.div(tf.square(x-mu), var))

        return tf.reduce_sum(result, -1, keep_dims=True)
