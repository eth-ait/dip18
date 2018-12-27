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
import os
import json
import pickle
import importlib

from constants import Constants
C = Constants()


class Configuration(object):
    """
    Configuration class to setup experiments.
    """

    KNOWN_DATA_PATHS = {'local':  "./data",  # TODO enter path to data here
                        'server': "./data"}  # TODO enter path to data here

    def __init__(self, **kwargs):
        self.config = kwargs

        if not("seed" in self.config):
            self.config['seed'] = C.SEED

        models = importlib.import_module('tf_models')
        self.model_cls = getattr(models, self.config.get('model_cls'))
        datasets = importlib.import_module('dataset')
        self.dataset_cls = getattr(datasets, self.config.get('dataset_cls'))

        self.data_dir = None
        self.set_data_dir()

    def set_data_dir(self):
        """
        Depending on the system and data_type sets dataset paths for training and validation.
        """
        try:
            self.data_dir = self.KNOWN_DATA_PATHS[self.config['system']]
        except KeyError:
            raise ValueError("system '{}' unknown, choose from {}".format(self.config['system'],
                                                                          ','.join(self.KNOWN_DATA_PATHS.keys())))

        df = self.config['data_file']

        # for training
        self.config['training_data'] = os.path.join(self.data_dir, "imu_{}_training.npz".format(df))
        self.config['validation_data'] = os.path.join(self.data_dir, "imu_{}_validation.npz".format(df))

        # for evaluation
        self.config['test_our_data'] = os.path.join(self.data_dir, "imu_own_test.npz")
        self.config['test_total_capture'] = os.path.join(self.data_dir, "imu_v9_test_total_capture.npz")
        self.config['test_playground_data'] = os.path.join(self.data_dir, "imu_v9_test_playground.npz")

    def get(self, param, default=None):
        """
        Get the value of the configuration parameter `param`. If `default` is set, this will be returned in case `param`
        does not exist. If `default` is not set and `param` does not exist, an error is thrown.
        """
        return self.config.get(param, default)

    def set(self, param, value, override=False):
        """
        Sets a new configuration parameter or overrides an existing one.
        """
        if not override and self.exists(param):
            raise RuntimeError('Key "{}" already exists. If you want to override set "override" to True.'.format(param))
        self.config[param] = value

    def exists(self, param):
        """
        Check if given configuration parameter exists.
        """
        return param in self.config.keys()

    def dump(self, path):
        """
        Stores this configuration object on disk in a human-readable format (json) and as a byte object (pickle).
        """
        pickle.dump(self.config, open(os.path.join(path, 'config.pkl'), 'wb'), protocol=2)
        json.dump(self.config, open(os.path.join(path, 'config.json'), 'w'), indent=4, sort_keys=True)

    def override_data_path(self, training, validation=None):
        """
        Override path to training and validation data sets. Both data sets are expected to be stored in `data_dir`.
        """
        self.config['training_data'] = os.path.join(self.data_dir, training)
        if validation is not None:
            self.config['validation_data'] = os.path.join(self.data_dir, validation)

    def get_preprocessing_ops(self):
        """
        Returns a dictionary specifying whether or not to apply certain pre-processing operations, like e.g.
        zero-mean unit-variance normalization.
        """
        return {C.PP_IMU_ORI_NORM: self.config.get(C.PP_IMU_ORI_NORM, False),
                C.PP_IMU_ACC_NORM: self.config.get(C.PP_IMU_ACC_NORM, False),
                C.PP_IMU_SMPL_NORM: self.config.get(C.PP_IMU_SMPL_NORM, False)}

    def get_sample_function(self):
        """
        Returns data-dependent sample construction functions (in tensorflow and numpy) given model outputs.
        For now we directly use model mean predictions.
        """
        def sample_np(out_dict):
            return out_dict[C.OUT_MU]

        def sample_tf(out_dict):
            return out_dict[C.OUT_MU]

        return sample_tf, sample_np

    @staticmethod
    def from_json(path):
        """
        Loads a configuration from json file.
        """
        return json.load(open(path, 'r'))

    @staticmethod
    def from_template(data_type, model_type):
        """
        Creates a configuration dictionary by using templates.
        """
        if data_type == C.IMU:
            data_config = Configuration.imu_data_config()
        else:
            raise Exception(data_type + " is not found.")

        if model_type == C.MODEL_RNN:
            model_config = Configuration.rnn_model_config()
        elif model_type == C.MODEL_BIRNN:
            model_config = Configuration.birnn_model_config()
        else:
            raise Exception(model_type + " is not found.")

        model_config['core_model_cls'] = model_config['model_cls']
        model_config['model_cls'] = "IMUWrapper"

        # We have two sources of hyper-parameters.
        # Priority order: model config_obj > data config_obj.
        return {**data_config, **model_config}

    @staticmethod
    def define_training_setup(parser):
        """
        Adds command line arguments for training script.

        Args:
            parser (argparse.ArgumentParser object):
        """
        parser.add_argument('--system', required=True, choices=list(Configuration.KNOWN_DATA_PATHS.keys()), type=str,
                            help='determines location of data and output paths')

        # Experiment outputs.
        parser.add_argument('--save_dir', default='./runs/',
                            help='Path to main model save directory.')
        parser.add_argument('--eval_dir', default='./runs_evaluation/',
                            help='Path to main log/output directory.')
        parser.add_argument('--checkpoint_id', default=None, help='Log and output directory.')
        parser.add_argument('--analyze_after_training', action="store_true", required=False,
                            help='Run evaluation after training.')

        # Load existing models or re-create existing models.
        parser.add_argument('--model_id', required=False, default=None,
                            help='Continue training with this model.')
        parser.add_argument('--model_type', required=False, choices=[C.MODEL_RNN, C.MODEL_BIRNN],
                            help='Determines model architecture.')
        parser.add_argument('--json_file', type=str, help='Creates a model exactly how configured in the json file.')

        # Data management.
        parser.add_argument('--data_file', required=False, default='v9', choices=['v9', 'own'],
                            help='Name of dataset file: v9 for AMASS, own for DIP-IMU')
        parser.add_argument('--' + C.PP_IMU_ORI_NORM, action="store_true",
                            help='Applies zero-mean unit-variance normalization on orientation.')
        parser.add_argument('--' + C.PP_IMU_ACC_NORM, action="store_true",
                            help='Applies zero-mean unit-variance normalization on acceleration.')
        parser.add_argument('--' + C.PP_IMU_SMPL_NORM, action="store_true",
                            help='Applies zero-mean unit-variance normalization on smple pose.')

        # Hyper-parameters.
        parser.add_argument('--use_acc_loss', action="store_true",
                            help='Auxiliary acceleration loss.')
        parser.add_argument('--use_ori_loss', action="store_true",
                            help='Reconstruct input orientations.')
        parser.add_argument('--ignore_acc_input', action="store_true",
                            help='Remove accelerations from input entirely.')

        # Fine-tuning.
        parser.add_argument('--finetune_train_data', default=None,
                            help='Name of *.npz file on which to fine-tune.')
        parser.add_argument('--finetune_valid_data', default=None,
                            help="Name of *.npz file on which to evaluate during fine-tuning.")

    @staticmethod
    def define_evaluation_setup(parser):
        """
        Adds command line arguments for evaluation script.

        Args:
            parser (argparse.ArgumentParser object):
        """
        parser.add_argument('--system', required=True, choices=list(Configuration.KNOWN_DATA_PATHS.keys()), type=str,
                            help='determines location of data and output paths')
        parser.add_argument('--data_file', required=False, default='v9', choices=['v9', 'own'],
                            help='Name of dataset file: v9 for AMASS, own for DIP-IMU')

        # Experiment outputs.
        parser.add_argument('--save_dir', type=str, default='./runs/',
                            help='Path to main model save directory.')
        parser.add_argument('--eval_dir', type=str,
                            help='Path to main log/output directory.')
        parser.add_argument('--model_id', required=True, type=str,
                            help='ID of model to be evaluated.')
        parser.add_argument('--checkpoint_id', type=str, default=None,
                            help='Model checkpoint. If not set, then the last checkpoint is used.')

        # On which dataset to evaluate.
        parser.add_argument('--datasets', nargs='+', choices=['dip-imu', 'tc', 'playground'],
                            help='On which dataset(s) to evaluate the given model. The respective test splits are'
                                 'chosen automatically.')

        # Verbosity.
        parser.add_argument('--verbose', dest='verbose', type=int, default=0,
                            help='Verbosity of logs.')
        parser.add_argument('--save_predictions', action="store_true",
                            help='Whether to save predictions as pkl files in quantitative mode.')

        # BiRNN offline or online params.
        parser.add_argument('--birnn_eval_chunks', action="store_true",
                            help='Whether to use shorter input chunks with BiRNN')
        parser.add_argument('--past_frames', nargs='+', default=[-1], type=int,
                            help='List of past frames. Used with --past_frames. If both are -1, then the '
                                 'whole sequence is used (i.e., offline mode.)')
        parser.add_argument('--future_frames', nargs='+', default=[-1], type=int,
                            help='List of future frames. Used with --past_frames. If both are -1, then the '
                                 'whole sequence is used (i.e., offline mode.)')

    def set_experiment_name(self, use_template=True, experiment_name=None):
        """
        Creates a folder name based on data and model configuration.

        Args:
            use_template (bool): Whether to use data and model naming template.
            experiment_name (str): A descriptive experiment name. It is used as prefix if use_template is True.

        Returns:
            A descriptive string to be used for experiment folder name.
        """
        if use_template:
            if self.config['model_type'] in [C.MODEL_RNN, C.MODEL_BIRNN]:
                str_drop = ""
                if self.config['input_layer'].get('dropout_rate', 0) > 0:
                    str_drop += "-idrop" + str(int(self.config['input_layer']['dropout_rate']*10))

                inp_fc_str = ""
                if self.config['input_layer']['num_layers'] > 0:
                    inp_fc_str = "-fc{}_{}".format(self.config['input_layer']['num_layers'],
                                                   self.config['input_layer']['size'])
                out_fc_str = ""
                if self.config['output_layer']['num_layers'] > 0:
                    out_fc_str = "-fc{}_{}".format(self.config['output_layer']['num_layers'],
                                                   self.config['output_layer']['size'])
                self.config['experiment_name'] = "{}{}-{}{}_{}{}{}-{}".format(
                    self.config['model_type'],
                    inp_fc_str, self.config['rnn_layer']['cell_type'],
                    self.config['rnn_layer']['num_layers'],
                    self.config['rnn_layer']['size'],
                    out_fc_str,
                    str_drop,
                    self.config['input_layer']['activation_fn'])

            self.config['experiment_name'] = self.config['data_file'] + "-" + self.config['experiment_name']

            str_pp = ""
            if self.config.get(C.PP_IMU_ORI_NORM, False):
                str_pp += "_ori"
            if self.config.get(C.PP_IMU_ACC_NORM, False):
                str_pp += "_acc"
            if self.config.get(C.PP_IMU_SMPL_NORM, False):
                str_pp += "_smpl"

            if str_pp != "":
                str_pp = "norm" + str_pp

            str_aux_loss = ""
            if self.config.get("use_acc_loss", False):
                str_aux_loss += "_acc"
            if self.config.get("use_ori_loss", False):
                str_aux_loss += "_ori"
            if str_aux_loss != "":
                str_aux_loss = "auxloss" + str_aux_loss
            str_pp += "" if str_aux_loss == "" else "-" + str_aux_loss
            if self.config.get('use_smpl_input', False):
                str_pp += "smpl_input" if str_pp == "" else "-smpl_input"

            self.config['experiment_name'] += str_pp

        else:
            self.config['experiment_name'] = ""

        if experiment_name is not None:
            self.config['experiment_name'] = experiment_name + "-" + self.config['experiment_name']

        return self.config['experiment_name']

    @staticmethod
    def imu_data_config(config=None):
        """
        Configuration for IMU data set.
        """
        config = config or {}

        config['model_save_dir'] = './runs_imu/'
        config['validate_model'] = True  # Whether to monitor validation loss during training or not
        config['use_staging_area'] = False  # Transfers data to GPU. Likely to improve performance.

        config['learning_rate'] = 1e-4
        config['learning_rate_type'] = 'exponential'  # 'fixed'  # 'exponential'
        config['learning_rate_decay_steps'] = 2000
        config['learning_rate_decay_rate'] = 0.96

        config['checkpoint_every_step'] = 1000  # store checkpoint aver this many iterations
        config['evaluate_every_step'] = 500  # validation and/or test performance
        config['print_every_step'] = 50  # print
        config['tensorboard_verbose'] = 1  # 1 for latent space scalar summaries, 2 for histogram summaries (debugging).

        config['reduce_loss'] = C.R_MEAN_STEP
        config['batch_size'] = 16
        config['num_epochs'] = 80

        config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
        config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.

        config['use_bucket_feeder'] = False
        config['dataset_cls'] = 'ImuDatasetTF'

        # Hidden layer.
        config['output_layer'] = {}
        config['output_layer']['num_layers'] = 1  # number of FC layers.
        config['output_layer']['size'] = 256  # number of FC neurons.
        config['output_layer']['activation_fn'] = C.RELU  # type of activation function after each FC layer.

        # Predictions, i.e., outputs of the model.
        config['output_layer']['out_keys'] = []  # [C.OUT_MU], defined through loss function
        config['output_layer']['out_dims'] = []  # Then dataset.target_dims will be used.
        config['output_layer']['out_activation_fn'] = []  # defined through loss function

        # Loss.
        nll_loss = dict()
        nll_loss['type'] = C.NLL_NORMAL
        nll_loss['out_key'] = "out"  # Looks for out_mu and out_sigma in model.output_ops
        nll_loss['weight'] = 1
        nll_loss['target_idx'] = 0
        config['loss'] = {"smpl": nll_loss}

        # Use SMPL output of previous time step as input.
        config['use_smpl_input'] = False

        return config

    @staticmethod
    def rnn_model_config(config=None):
        """
        Configuration of RNN model.
        """
        config = config or {}

        config['model_cls'] = 'RNNAutoRegressive'

        config['input_layer'] = {}
        if config['input_layer'] == {}:
            config['input_layer']['dropout_rate'] = 0  # Dropout rate on inputs directly.
            config['input_layer']['num_layers'] = 1  # number of fully connected (FC) layers on top of RNN.
            config['input_layer']['size'] = 512  # number of FC neurons.
            config['input_layer']['activation_fn'] = C.RELU  # type of activation function after each FC layer.

        config['rnn_layer'] = {}  # See get_rnn_cell function in tf_model_utils.
        config['rnn_layer']['num_layers'] = 2  # (default: 1)
        config['rnn_layer']['cell_type'] = C.LSTM  # (default: 'lstm')
        config['rnn_layer']['size'] = 512  # (default: 512)

        # Override training hyper-parameters.
        config['grad_clip_by_norm'] = 1  # If it is 0, then gradient clipping will not be applied.
        config['grad_clip_by_value'] = 0  # If it is 0, then gradient clipping will not be applied.
        config['print_every_step'] = 1

        return config

    @staticmethod
    def birnn_model_config(config=None):
        config = Configuration.rnn_model_config(config)
        config['model_cls'] = 'BiRNN'
        return config
