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

The model is trained by using negative log-likelihood (reconstruction) and KL-divergence losses.
Assuming that model outputs are isotropic Gaussian distributions.

Model functionality is decomposed into basic functions (see build_graph method) so that variants of the model can easily
be constructed by inheriting from this vanilla architecture.

Note that different modes (i.e., training, validation, sampling) should be implemented as different graphs by reusing
the parameters. Therefore, validation functionality shouldn't be used by a model with training mode.
"""
import tensorflow as tf
import numpy as np
import sys
import time
import math
import copy

import tf_model_utils
from tf_model_utils import get_reduce_loss_func, get_rnn_cell, linear, fully_connected_layer, get_activation_fn
from constants import Constants

C = Constants()


class BaseTemporalModel(object):
    """
    Model class for modeling of temporal data, providing auxiliary functions implementing tensorflow routines and
    abstract functions to build model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats):
        self.config = config

        # "sampling" is only valid for generative models.
        assert mode in ["training", "validation", "test", "sampling"]
        self.mode = mode
        self.is_sampling = mode == "sampling"
        self.is_validation = mode == "validation" or mode == "test"
        self.is_training = mode == "training"
        self.print_every_step = self.config.get('print_every_step')

        self.reuse = reuse
        self.session = session

        self.placeholders = placeholders
        self.pl_inputs = placeholders[C.PL_INPUT]
        self.pl_targets = placeholders[C.PL_TARGET]
        self.pl_seq_length = placeholders[C.PL_SEQ_LEN]
        self.seq_loss_mask = tf.expand_dims(tf.sequence_mask(lengths=self.pl_seq_length, dtype=tf.float32), -1)

        # Creates a sample by using model outputs.
        self.sample_fn_tf, self.sample_fn_np = config.get_sample_function()

        self.input_dims = input_dims.copy()
        self.target_dims = target_dims.copy()
        self.target_pieces = tf.split(self.pl_targets, target_dims, axis=2)

        self.norm_smpl = self.config.get("norm_smpl", False)
        if self.is_sampling and data_stats is not None:
            self.data_stats = data_stats["smpl_pose"].tolist()
            self.data_mean = self.data_stats["mean_channel"]
            self.data_std = self.data_stats["std_channel"]

            self.data_stats_ori = data_stats["orientation"].tolist()
            self.data_mean_ori = self.data_stats_ori["mean_channel"]
            self.data_std_ori = self.data_stats_ori["std_channel"]

            self.data_stats_acc = data_stats["acceleration"].tolist()
            self.data_mean_acc = self.data_stats_acc["mean_channel"]
            self.data_std_acc = self.data_stats_acc["std_channel"]

        input_shape = self.pl_inputs.shape.as_list()
        self.batch_size = input_shape[0]
        self.sequence_length = -1 if input_shape[1] is None else input_shape[1]

        self.output_layer_config = copy.deepcopy(config.get('output_layer'))

        # Update output ops.
        self.nll_loss_config = copy.deepcopy(self.config.get('nll_loss', None))  # legacy version
        self.loss_config = copy.deepcopy(self.config.get('loss', None))

        if self.nll_loss_config is not None:
            if self.output_layer_config['out_dims'] is None:
                self.output_layer_config['out_dims'] = self.target_dims
            else:
                assert self.output_layer_config['out_dims'] == self.target_dims, "Output layer dimensions don't " \
                                                                                 "match with dataset target dimensions."

            # Assuming that each loss is represented only once and C.OUT_MU is assigned to the following loss types:
            out_mu_index = self.output_layer_config['out_keys'].index(C.OUT_MU)

            if self.nll_loss_config['type'][out_mu_index] in [C.NLL_NORMAL]:
                self.output_layer_config['out_keys'].append(C.OUT_SIGMA)
                self.output_layer_config['out_dims'].append(self.output_layer_config['out_dims'][out_mu_index])
                self.output_layer_config['out_activation_fn'].append(C.SOFTPLUS)
        else:
            for loss_name, loss_entry in self.loss_config.items():
                self.define_loss(loss_entry)

        # Ops to be evaluated by training loop function. It is a dictionary containing <key, value> pairs where the
        # `value` is tensorflow graph op. For example, summary, loss, training operations. Note that different modes
        # (i.e., training, sampling, validation) may have different set of ops.
        self.ops_run_loop = dict()
        # `summary` ops are kept in a list.
        self.ops_run_loop['summary'] = []

        # Dictionary of model outputs such as logits or mean and sigma of Gaussian distribution modeling outputs.
        # They are used in making predictions and creating loss terms.
        self.ops_model_output = {}

        # To keep track of loss ops. List of loss terms that must be evaluated by session.run during training.
        self.ops_loss = {}

        # (Default) graph ops to be fed into session.run while evaluating the model. Note that tf_evaluate* codes expect
        # to get these op results.
        self.ops_evaluation = {}

        # Graph ops for scalar summaries such as average predicted variance.
        self.ops_scalar_summary = {}

    def define_loss(self, loss_config):
        if loss_config['type'] in [C.NLL_NORMAL]:
            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_MU)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(None)

            self.output_layer_config['out_keys'].append(loss_config['out_key'] + C.SUF_SIGMA)
            self.output_layer_config['out_dims'].append(self.target_dims[loss_config['target_idx']])
            self.output_layer_config['out_activation_fn'].append(C.SOFTPLUS)

    def build_graph(self):
        """
        Called by TrainingEngine. Assembles modules of tensorflow computational graph by creating model, loss terms and
        summaries for tensorboard. Applies preprocessing on the inputs and postprocessing on model outputs if necessary.
        """
        raise NotImplementedError('subclasses must override build_graph method')

    def build_network(self):
        """
        Builds internal dynamics of the model. Sets
        """
        raise NotImplementedError('subclasses must override build_network method')

    def sample(self, **kwargs):
        """
        Draws samples from model.
        """
        raise NotImplementedError('subclasses must override sample method')

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        raise NotImplementedError('subclasses must override reconstruct method')

    def build_loss_terms(self):
        """
        Builds loss terms.
        """
        # Function to get final loss value, i.e., average or sum.
        self.reduce_loss_fn = get_reduce_loss_func(self.config.get('reduce_loss'),
                                                   tf.reduce_sum(self.seq_loss_mask, axis=[1, 2]))
        # Legacy mode
        if self.nll_loss_config is not None:
            for idx, loss_type in enumerate(self.nll_loss_config['type']):
                loss_key = "loss_" + loss_type
                if loss_key not in self.ops_loss:
                    with tf.name_scope(loss_type):
                        # Negative log likelihood loss.
                        if loss_type == C.NLL_NORMAL:
                            logli_term = tf_model_utils.logli_normal_isotropic(self.target_pieces[idx],
                                                                               self.ops_model_output[C.OUT_MU],
                                                                               self.ops_model_output[C.OUT_SIGMA])
                        elif loss_type == C.MSE:
                            logli_term = -tf.reduce_sum(
                                tf.square(self.target_pieces[idx] - self.ops_model_output[C.OUT_MU]), axis=2,
                                keepdims=True)
                        else:
                            raise Exception(loss_type + " is not implemented.")

                        nll_loss_term = -self.nll_loss_config['weight'][idx]*self.reduce_loss_fn(
                            self.seq_loss_mask*logli_term)
                        self.ops_loss[loss_key] = nll_loss_term
        else:
            for loss_name, loss_entry in self.loss_config.items():
                loss_type = loss_entry['type']
                out_key = loss_entry['out_key']
                target_idx = loss_entry['target_idx']
                loss_key = loss_name + "_" + loss_type
                if loss_key not in self.ops_loss:
                    with tf.name_scope(loss_key):
                        # Negative log likelihood loss.
                        if loss_type == C.NLL_NORMAL:
                            logli_term = tf_model_utils.logli_normal_isotropic(self.target_pieces[target_idx],
                                                                               self.ops_model_output[
                                                                                   out_key + C.SUF_MU],
                                                                               self.ops_model_output[out_key + C.SUF_SIGMA])
                        elif loss_type == C.MSE:
                            logli_term = -tf.reduce_sum(
                                tf.square(self.target_pieces[target_idx] - self.ops_model_output[out_key + C.SUF_MU]),
                                axis=2, keepdims=True)
                        else:
                            raise Exception(loss_type + " is not implemented.")

                        loss_term = -loss_entry['weight']*self.reduce_loss_fn(self.seq_loss_mask*logli_term)
                        self.ops_loss[loss_key] = loss_term

    def build_total_loss(self):
        """
        Accumulate losses to create training optimization. Model.loss is used by the optimization function.
        """
        self.loss = 0
        for _, loss_op in self.ops_loss.items():
            self.loss += loss_op
        self.ops_loss['total_loss'] = self.loss

    def build_summary_plots(self):
        """
        Creates scalar summaries for loss plots. Iterates through `ops_loss` member and create a summary entry.

        If the model is in `validation` mode, then we follow a different strategy. In order to have a consistent
        validation report over iterations, we first collect model performance on every validation mini-batch
        and then report the average loss. Due to tensorflow's lack of loss averaging ops, we need to create
        placeholders per loss to pass the average loss.
        """
        if self.is_training:
            # For each loss term, create a tensorboard plot.
            for loss_name, loss_op in self.ops_loss.items():
                tf.summary.scalar(loss_name, loss_op, collections=[self.mode + '_summary_plot', self.mode + '_loss'])

        else:
            # Validation: first accumulate losses and then plot.
            # Create containers and placeholders for every loss term. After each validation step, keeps summing losses.
            # At the end of validation loop, calculates average performance on the whole validation dataset and creates
            # summary entries.
            self.container_loss = dict()
            self.container_loss_placeholders = dict()
            self.container_loss_summaries = dict()
            self.container_validation_feed_dict = dict()
            self.validation_summary_num_runs = 0

            for loss_name, _ in self.ops_loss.items():
                self.container_loss[loss_name] = 0
                self.container_loss_placeholders[loss_name] = tf.placeholder(tf.float32, shape=[])
                tf.summary.scalar(loss_name, self.container_loss_placeholders[loss_name],
                                  collections=[self.mode + '_summary_plot', self.mode + '_loss'])
                self.container_validation_feed_dict[self.container_loss_placeholders[loss_name]] = 0.0

        for summary_name, scalar_summary_op in self.ops_scalar_summary.items():
            tf.summary.scalar(summary_name, scalar_summary_op,
                              collections=[self.mode + '_summary_plot', self.mode + '_scalar_summary'])

    def finalise_graph(self):
        """
        Finalises graph building. It is useful if child classes must create some ops first.
        """
        self.loss_summary = tf.summary.merge_all(self.mode + '_summary_plot')
        if self.is_training:
            self.register_run_ops('summary', self.loss_summary)

        self.register_run_ops('loss', self.ops_loss)

    def training_step(self, step, epoch, feed_dict=None):
        """
        Training loop function. Takes a batch of samples, evaluates graph ops and updates model parameters.

        Args:
            step: current step.
            epoch: current epoch.
            feed_dict (dict): feed dictionary.

        Returns (dict): evaluation results.
        """
        start_time = time.perf_counter()
        ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)

        if math.isnan(ops_run_loop_results['loss']['total_loss']):
            raise Exception("nan values.")

        if step%self.print_every_step == 0:
            time_elapsed = (time.perf_counter() - start_time)
            self.log_loss(ops_run_loop_results['loss'], step, epoch, time_elapsed, prefix=self.mode + ": ")

        return ops_run_loop_results

    def evaluation_step(self, step, epoch, num_iterations, feed_dict=None):
        """
        Evaluation loop function. Evaluates the whole validation/test dataset and logs performance.

        Args:
            step: current step.
            epoch: current epoch.
            feed_dict (dict): feed dictionary.

        Returns: summary object.
        """
        start_time = time.perf_counter()
        for i in range(num_iterations):
            ops_run_loop_results = self.session.run(self.ops_run_loop, feed_dict=feed_dict)
            self.update_validation_loss(ops_run_loop_results['loss'])

        summary, total_loss = self.get_validation_summary()
        loss_out = total_loss['total_loss']

        time_elapsed = (time.perf_counter() - start_time)
        self.log_loss(total_loss, step, epoch, time_elapsed, prefix=self.mode + ": ")
        self.reset_validation_loss()

        return summary, loss_out

    def log_loss(self, eval_loss, step=0, epoch=0, time_elapsed=None, prefix=""):
        """
        Prints status messages during training. It is called in the main training loop.
        Args:
            eval_loss (dict): evaluated results of `ops_loss` dictionary.
            step (int): current step.
            epoch (int): current epoch.
            time_elapsed (float): elapsed time.
            prefix (str): some informative text. For example, "training" or "validation".
        """
        loss_format = prefix + "{}/{} \t Total: {:.4f} \t"
        loss_entries = [step, epoch, eval_loss['total_loss']]

        for loss_key in sorted(eval_loss.keys()):
            if loss_key != 'total_loss':
                loss_format += "{}: {:.4f} \t"
                loss_entries.append(loss_key)
                loss_entries.append(eval_loss[loss_key])

        if time_elapsed is not None:
            print(loss_format.format(*loss_entries) + "time/batch = {:.3f}".format(time_elapsed))
        else:
            print(loss_format.format(*loss_entries))

    def register_run_ops(self, op_key, op):
        """
        Adds a new graph op into `self.ops_run_loop`.

        Args:
            op_key (str): dictionary key.
            op: tensorflow op

        Returns:
        """
        if op_key in self.ops_run_loop and isinstance(self.ops_run_loop[op_key], list):
            self.ops_run_loop[op_key].append(op)
        else:
            self.ops_run_loop[op_key] = op

    def flat_tensor(self, tensor, dim=-1):
        """
        Reshapes a tensor such that it has 2 dimensions. The dimension specified by `dim` is kept.
        """
        keep_dim_size = tensor.shape.as_list()[dim]
        return tf.reshape(tensor, [-1, keep_dim_size])

    def temporal_tensor(self, flat_tensor):
        """
        Reshapes a flat tensor (2-dimensional) to a tensor with shape (batch_size, seq_len, feature_size). Assuming
        that the flat tensor has shape of (batch_size*seq_len, feature_size).
        """
        feature_size = flat_tensor.shape.as_list()[1]
        return tf.reshape(flat_tensor, [self.batch_size, -1, feature_size])

    def log_num_parameters(self):
        """
        Prints total number of parameters.
        """
        num_param = 0
        for v in tf.global_variables():
            num_param += np.prod(v.shape.as_list())

        self.num_parameters = num_param
        print("# of parameters: " + str(num_param))
        self.config.set('total_parameters', int(self.num_parameters), override=True)

    ########################################
    # Summary methods for validation mode.
    ########################################
    def update_validation_loss(self, loss_evaluated):
        """
        Updates validation losses. Note that this method is called after every validation step.

        Args:
            loss_evaluated: valuated results of `ops_loss` dictionary.
        """
        self.validation_summary_num_runs += 1
        for loss_name, loss_value in loss_evaluated.items():
            self.container_loss[loss_name] += loss_value

    def reset_validation_loss(self):
        """
        Resets validation loss containers.
        """
        for loss_name, loss_value in self.container_loss.items():
            self.container_loss[loss_name] = 0

    def get_validation_summary(self):
        """
        Creates a feed dictionary of validation losses for validation summary. Note that this method is called after
        validation loops is over.

        Returns (dict, dict):
            feed_dict for validation summary.
            average `ops_loss` results for `log_loss` method.
        """
        for loss_name, loss_pl in self.container_loss_placeholders.items():
            self.container_loss[loss_name] /= self.validation_summary_num_runs
            self.container_validation_feed_dict[loss_pl] = self.container_loss[loss_name]

        self.validation_summary_num_runs = 0
        valid_summary = self.session.run(self.loss_summary, self.container_validation_feed_dict)
        return valid_summary, self.container_loss


class BaseRNN(BaseTemporalModel):
    """
    Implements abstract build_graph and build_network methods to build an RNN model.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats):
        super(BaseRNN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats)

        self.input_layer_config = config.get('input_layer')
        self.rnn_layer_config = config.get('rnn_layer')

    def build_graph(self):
        """
        Builds model and creates plots for tensorboard. Decomposes model building into sub-modules and makes inheritance
        is easier.
        """
        self.build_network()
        self.build_loss_terms()
        self.build_total_loss()
        self.build_summary_plots()
        self.finalise_graph()
        if self.reuse is False:
            self.log_num_parameters()

    def build_network(self):
        self.build_cell()
        self.build_input_layer()
        self.build_rnn_layer()
        self.build_output_layer()

    def build_cell(self):
        """
        Builds a Tensorflow RNN cell object by using the given configuration `self.rnn_layer_config`.
        """
        self.cell = get_rnn_cell(scope='rnn_cell', reuse=self.reuse, **self.rnn_layer_config)
        self.initial_states = self.cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

    def build_input_layer(self):
        """
        Builds a number fully connected layers projecting the inputs into an intermediate representation  space.
        """
        if self.input_layer_config is not None:
            with tf.variable_scope('input_layer', reuse=self.reuse):
                if self.input_layer_config.get("dropout_rate", 0) > 0:
                    self.inputs_hidden = tf.layers.dropout(self.pl_inputs,
                                                           rate=self.input_layer_config.get("dropout_rate"),
                                                           noise_shape=None,
                                                           seed=17,
                                                           training=self.is_training)
                else:
                    self.inputs_hidden = self.pl_inputs

                if self.input_layer_config.get("num_layers", 0) > 0:
                    flat_inputs_hidden = self.flat_tensor(self.inputs_hidden)
                    flat_inputs_hidden = fully_connected_layer(flat_inputs_hidden, **self.input_layer_config)
                    self.inputs_hidden = self.temporal_tensor(flat_inputs_hidden)
        else:
            self.inputs_hidden = self.pl_inputs

    def build_rnn_layer(self):
        """
        Builds RNN layer by using dynamic_rnn wrapper of Tensorflow.
        """
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_output_state = tf.nn.dynamic_rnn(self.cell,
                                                                        self.inputs_hidden,
                                                                        sequence_length=self.pl_seq_length,
                                                                        initial_state=self.initial_states,
                                                                        dtype=tf.float32)
            self.output_layer_inputs = self.rnn_outputs
            self.ops_evaluation['state'] = self.rnn_output_state

    def build_output_layer(self):
        """
        Builds a number fully connected layers projecting RNN predictions into an embedding space. Then, for each model
        output is predicted by a linear layer.
        """
        flat_outputs_hidden = self.flat_tensor(self.output_layer_inputs)
        with tf.variable_scope('output_layer_hidden', reuse=self.reuse):
            flat_outputs_hidden = fully_connected_layer(flat_outputs_hidden, is_training=self.is_training, **self.output_layer_config)

        for idx in range(len(self.output_layer_config['out_keys'])):
            key = self.output_layer_config['out_keys'][idx]

            with tf.variable_scope('output_layer_' + key, reuse=self.reuse):
                flat_out = linear(input_=flat_outputs_hidden,
                                  output_size=self.output_layer_config['out_dims'][idx],
                                  activation_fn=self.output_layer_config['out_activation_fn'][idx],
                                  is_training=self.is_training)

                self.ops_model_output[key] = self.temporal_tensor(flat_out)

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def inference_step(self, orientations, accelerations, previous_state, **kwargs):
        input_sequence = np.concatenate([orientations, accelerations], axis=-1)

        if self.is_sampling:
            data_mean = np.concatenate([self.data_mean_ori, self.data_mean_acc], axis=-1)
            data_std = np.concatenate([self.data_std_ori, self.data_std_acc], axis=-1)
            input_sequence = (input_sequence - data_mean) / data_std

        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = {self.pl_inputs: input_sequence,
                     self.pl_seq_length: np.array([input_sequence.shape[1]])}

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs["loss"])

        if self.is_sampling and self.norm_smpl:
            model_outputs["sample"] = model_outputs["sample"]*self.data_std + self.data_mean

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]
            input_sequence = input_sequence[0]

        return model_outputs["sample"], model_outputs['state'], input_sequence


class RNNAutoRegressive(BaseRNN):
    """
    Auto-regressive RNN model. Predicts next step (t+1) given the current step (t). Note that here we assume targets are
    equivalent to inputs shifted by one step in time.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats):
        super(RNNAutoRegressive, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats)

    def build_output_layer(self):
        # Prediction layer.
        BaseRNN.build_output_layer(self)

        num_entries = tf.cast(tf.reduce_sum(self.pl_seq_length), tf.float32)
        self.ops_scalar_summary["mean_out_mu"] = tf.reduce_sum(self.ops_model_output[C.OUT_MU] * self.seq_loss_mask) / (
                    num_entries * self.ops_model_output[C.OUT_MU].shape.as_list()[-1])
        if C.OUT_SIGMA in self.ops_model_output:
            self.ops_scalar_summary["mean_out_sigma"] = tf.reduce_sum(
                self.ops_model_output[C.OUT_SIGMA] * self.seq_loss_mask) / (num_entries * self.ops_model_output[
                C.OUT_SIGMA].shape.as_list()[-1])

        self.output_sample = self.sample_fn_tf(self.ops_model_output)
        self.input_sample = self.pl_inputs
        self.ops_evaluation['sample'] = self.output_sample

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get('input_sequence', None)
        target_sequence = kwargs.get('target_sequence', None)

        if self.is_sampling:
            data_mean = np.concatenate([self.data_mean_ori, self.data_mean_acc], axis=-1)
            data_std = np.concatenate([self.data_std_ori, self.data_std_acc], axis=-1)
            input_sequence = (input_sequence - data_mean) / data_std

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = {self.pl_inputs: input_sequence,
                     self.pl_seq_length: np.array([input_sequence.shape[1]])}

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation['loss'] = self.ops_loss
            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs['loss'])

        if self.is_sampling and self.norm_smpl:
            model_outputs["sample"] = model_outputs["sample"] * self.data_std + self.data_mean

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def sample(self, **kwargs):
        """
        Sampling function.
        Args:
            **kwargs:
        """
        seed_sequence = kwargs.get('seed_sequence', None)
        sample_length = kwargs.get('sample_length', 100)

        assert seed_sequence is not None, "Need a seed sample."
        batch_dimension = seed_sequence.ndim == 3
        if batch_dimension is False:
            seed_sequence = np.expand_dims(seed_sequence, axis=0)

        # Feed seed sequence and update RNN state.
        if not("state" in self.ops_model_output):
            self.ops_evaluation["state"] = self.rnn_output_state
        model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: seed_sequence})

        # Get the last step.
        last_step = model_outputs['sample'][:, -1:, :]
        synthetic_sequence = self.sample_function(last_step, model_outputs['state'], sample_length)

        model_outputs = {}
        if batch_dimension is False:
            model_outputs["sample"] = synthetic_sequence[0]
        else:
            model_outputs["sample"] = synthetic_sequence

        return model_outputs

    def sample_function(self, current_input, previous_state, sample_length):
        """
        Auxiliary method to draw sequence of samples in auto-regressive fashion.
        Args:
        Returns:
            Synthetic samples as numpy array (batch_size, sample_length, feature_size)
        """
        sequence = current_input.copy()
        for step in range(sample_length):
            model_input = sequence[:,-1:,:]
            model_outputs = self.session.run(self.ops_evaluation, feed_dict={self.pl_inputs: model_input, self.initial_states:previous_state})
            previous_state = model_outputs['state']

            sequence = np.concatenate([sequence, model_outputs['sample']], axis=1)
        return sequence[:,-sample_length:]


class BiRNN(BaseRNN):
    """
    Bi-directional RNN.
    """
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats):
        super(BiRNN, self).__init__(config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats)

        self.cells_fw = []
        self.cells_bw = []

        self.initial_states_fw = []
        self.initial_states_bw = []

        # See https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/stack_bidirectional_dynamic_rnn
        self.stack_fw_bw_cells = self.rnn_layer_config.get('stack_fw_bw_cells', True)

    def build_cell(self):
        """
        Builds a Tensorflow RNN cell object by using the given configuration `self.rnn_layer_config`.
        """
        if self.stack_fw_bw_cells:
            single_cell_config = self.rnn_layer_config.copy()
            single_cell_config['num_layers'] = 1
            for i in range(self.rnn_layer_config['num_layers']):
                cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **single_cell_config)
                self.cells_fw.append(cell_fw)
                self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

                cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **single_cell_config)
                self.cells_bw.append(cell_bw)
                self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))
        else:
            cell_fw = get_rnn_cell(scope='rnn_cell_fw', reuse=self.reuse, **self.rnn_layer_config)
            self.cells_fw.append(cell_fw)
            self.initial_states_fw.append(cell_fw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

            cell_bw = get_rnn_cell(scope='rnn_cell_bw', reuse=self.reuse, **self.rnn_layer_config)
            self.cells_bw.append(cell_bw)
            self.initial_states_bw.append(cell_bw.zero_state(batch_size=self.batch_size, dtype=tf.float32))

    def build_rnn_layer(self):
        with tf.variable_scope("bidirectional_rnn_layer", reuse=self.reuse):
            if self.stack_fw_bw_cells:
                self.rnn_outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                                                            cells_fw=self.cells_fw,
                                                                            cells_bw=self.cells_bw,
                                                                            inputs=self.inputs_hidden,
                                                                            initial_states_fw=self.initial_states_fw,
                                                                            initial_states_bw=self.initial_states_bw,
                                                                            dtype=tf.float32,
                                                                            sequence_length=self.pl_seq_length)
            else:
                outputs_tuple, output_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.cells_fw[0],
                    cell_bw=self.cells_bw[0],
                    inputs=self.inputs_hidden,
                    sequence_length=self.pl_seq_length,
                    initial_state_fw=self.initial_states_fw[0],
                    initial_state_bw=self.initial_states_bw[0],
                    dtype=tf.float32)

                self.rnn_outputs = tf.concat(outputs_tuple, 2)
                self.output_state_fw, self.output_state_bw = output_states

            self.output_layer_inputs = self.rnn_outputs
            self.ops_evaluation["state"] = [self.output_state_fw, self.output_state_bw]

    def reconstruct(self, **kwargs):
        """
        Predicts the next step by using previous ground truth steps. If the target sequence is passed, then loss is also
        reported.
        Args:
            **kwargs:
        Returns:
            Predictions of next steps (batch_size, input_seq_len, feature_size)
        """
        input_sequence = kwargs.get("input_sequence", None)
        target_sequence = kwargs.get("target_sequence", None)

        if self.is_sampling:
            data_mean = np.concatenate([self.data_mean_ori, self.data_mean_acc], axis=-1)
            data_std = np.concatenate([self.data_std_ori, self.data_std_acc], axis=-1)
            input_sequence = (input_sequence - data_mean) / data_std

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        feed_dict = {self.pl_inputs: input_sequence,
                     self.pl_seq_length: np.array([input_sequence.shape[1]])}

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if "loss" not in self.ops_evaluation:
                self.ops_evaluation["loss"] = self.ops_loss

            feed_dict[self.pl_targets] = target_sequence

        model_outputs = self.session.run(self.ops_evaluation, feed_dict)
        if "loss" in model_outputs:
            self.log_loss(model_outputs["loss"])

        if self.is_sampling and self.norm_smpl:
            model_outputs["sample"] = model_outputs["sample"] * self.data_std + self.data_mean

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs

    def reconstruct_chunks(self, **kwargs):
        input_sequence = kwargs.get("input_sequence", None)
        target_sequence = kwargs.get("target_sequence", None)
        len_past = kwargs.get("len_past", None)
        len_future = kwargs.get("len_future", None)

        if self.is_sampling:
            data_mean = np.concatenate([self.data_mean_ori, self.data_mean_acc], axis=-1)
            data_std = np.concatenate([self.data_std_ori, self.data_std_acc], axis=-1)
            input_sequence = (input_sequence - data_mean) / data_std

        assert input_sequence is not None, "Need an input sample."
        batch_dimension = input_sequence.ndim == 3
        if batch_dimension is False:
            input_sequence = np.expand_dims(input_sequence, axis=0)

        if target_sequence is not None:
            if batch_dimension is False:
                target_sequence = np.expand_dims(target_sequence, axis=0)

            if not "loss" in self.ops_evaluation:
                self.ops_evaluation["loss"] = self.ops_loss

        batch_size, seq_len, input_size = input_sequence.shape

        predictions = []
        loss = 0.0
        feed_dict = {}

        for step in range(seq_len):
            start_idx = max(step-len_past, 0)
            end_idx = min(step+len_future+1, seq_len)
            print('\rprocessing step {}/{} from {} to {}'.format(step+1, seq_len, start_idx, end_idx), end='')

            feed_dict[self.pl_inputs] = input_sequence[:, start_idx:end_idx]
            feed_dict[self.pl_seq_length] = np.array([end_idx-start_idx]*batch_size)
            if target_sequence is not None:
                feed_dict[self.pl_targets] = target_sequence[:, start_idx:end_idx]

            model_outputs = self.session.run(self.ops_evaluation, feed_dict)
            prediction_step = min(step, len_past)
            predictions.append(model_outputs["sample"][:, prediction_step:prediction_step+1])
            if "loss" in model_outputs:
                loss += model_outputs["loss"]["total_loss"]*batch_size

        model_outputs = dict()
        model_outputs["sample"] = np.concatenate(predictions, axis=1)
        model_outputs["loss"] = {"total_loss": loss/(batch_size*seq_len)}

        if self.is_sampling and self.norm_smpl:
            model_outputs["sample"] = model_outputs["sample"] * self.data_std + self.data_mean

        if batch_dimension is False:
            model_outputs["sample"] = model_outputs["sample"][0]

        return model_outputs


class IMUWrapper(object):
    def __init__(self, config, session, reuse, mode, placeholders, input_dims, target_dims, data_stats):
        self.config = config

        ORI_SIZE, ACC_SIZE = 45, 15
        self.use_acc_loss = mode == "training" and self.config.get("use_acc_loss", False)
        self.use_ori_loss = mode == "training" and self.config.get("use_ori_loss", False)
        self.ignore_acc = config.get('ignore_acc_input')

        self.orientation, self.acceleration = tf.split(placeholders[C.PL_INPUT], [ORI_SIZE, ACC_SIZE], axis=2)
        if self.ignore_acc:
            print("IGNORING ACC INPUT")
            placeholders[C.PL_INPUT] = self.orientation
            self.use_acc_loss = False  # cannot use acc loss when no acceleration present

        self.smpl_pose = placeholders[C.PL_TARGET]

        self.input_dims = input_dims.copy()
        self.target_dims = target_dims.copy()
        original_loss_config = copy.deepcopy(config.config.get('loss', None))
        self.sess = session

        if self.use_ori_loss:
            placeholders[C.PL_TARGET] = tf.concat([placeholders[C.PL_TARGET], self.orientation], axis=-1)
            self.target_dims.append(ORI_SIZE)
            ori_loss = dict()
            ori_loss["type"] = config.config["loss"]["smpl"]["type"]
            ori_loss["out_key"] = "ori_out"  # Looks for ori_out_mu and ori_out_sigma in model.output_ops
            ori_loss["weight"] = 1
            ori_loss["target_idx"] = len(config.config['loss'])
            config.config['loss']["ori"] = ori_loss

        if self.use_acc_loss:
            placeholders[C.PL_TARGET] = tf.concat([placeholders[C.PL_TARGET], self.acceleration], axis=-1)
            self.target_dims.append(ACC_SIZE)
            acc_loss = dict()
            acc_loss["type"] = config.config["loss"]["smpl"]["type"]
            acc_loss["out_key"] = "acc_out"  # Looks for acc_out_mu and acc_out_sigma in model.output_ops
            acc_loss["weight"] = 1
            acc_loss["target_idx"] = len(config.config['loss'])
            config.config['loss']["acc"] = acc_loss

        model_cls = getattr(sys.modules[__name__], self.config.get('core_model_cls'))
        self.model = model_cls(config, session, reuse, mode, placeholders, self.input_dims, self.target_dims, data_stats)
        self.pl_inputs = self.model.pl_inputs

        self.use_smpl_input = self.config.get("use_smpl_input", False)
        if self.use_smpl_input and not self.model.is_sampling:
            identity_smpl = tf.tile(tf.constant([[[1., 0., 0., 0., 1., 0., 0., 0., 1.]*15]]), [self.model.batch_size, 1, 1])
            self.smpl_inputs = tf.concat([identity_smpl, self.smpl_pose[:, :-1]], axis=1)
            self.model.pl_inputs = tf.concat([self.model.pl_inputs, self.smpl_inputs], axis=2)
        elif self.use_smpl_input and self.model.is_sampling:
            # In sampling mode smpl_inputs is prediction from the previous step.
            self.model.pl_inputs = tf.concat([self.model.pl_inputs, self.smpl_pose], axis=2)

        if original_loss_config is not None:
            config.config['loss'] = original_loss_config

    def build_graph(self):
        self.model.build_graph()
        self.loss = self.model.loss
        self.ops_loss = self.model.ops_loss
        self.seq_loss_mask = self.model.seq_loss_mask
        self.pl_targets = self.model.pl_targets
        self.pl_seq_length = self.model.pl_seq_length
        self.output_sample = self.model.output_sample

    def inference_step(self, orientations, accelerations, previous_state, **kwargs):
        return self.model.inference_step(orientations, accelerations, previous_state, **kwargs)

    def sample(self, **kwargs):
        return self.model.sample(**kwargs)

    def reconstruct(self, **kwargs):
        if self.ignore_acc:
            input_sequence = kwargs.get("input_sequence", None)
            kwargs['input_sequence'] = input_sequence[:, :45]
        return self.model.reconstruct(**kwargs)

    def build_loss_terms(self):
        self.model.build_loss_terms()

    def build_total_loss(self):
        self.model.build_total_loss()

    def build_summary_plots(self):
        self.model.build_summary_plots()

    def finalise_graph(self):
        self.model.finalise_graph()

    def training_step(self, step, epoch, feed_dict=None):
        return self.model.training_step(step, epoch, feed_dict)

    def evaluation_step(self, step, epoch, num_iterations, feed_dict=None):
        return self.model.evaluation_step(step, epoch, num_iterations, feed_dict)

    def log_loss(self, eval_loss, step=0, epoch=0, time_elapsed=None, prefix=""):
        self.model.log_loss(eval_loss, step, epoch, time_elapsed, prefix)

    def register_run_ops(self, op_key, op):
        self.model.register_run_ops(op_key, op)

    def log_num_parameters(self):
        self.model.log_num_parameters()

    def update_validation_loss(self, loss_evaluated):
        self.model.update_validation_loss(loss_evaluated)

    def reset_validation_loss(self):
        self.model.reset_validation_loss()

    def get_validation_summary(self):
        return self.model.get_validation_summary()
