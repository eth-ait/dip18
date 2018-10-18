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


class Constants(object):
    SEED = 23

    # RNN cells
    GRU = 'gru'
    LSTM = 'lstm'
    LayerNormLSTM = 'LayerNormBasicLSTMCell'

    # Activation functions
    RELU = 'relu'
    ELU = 'elu'
    SIGMOID = 'sigmoid'
    SOFTPLUS = 'softplus'
    TANH = 'tanh'
    SOFTMAX = 'softmax'
    LRELU = 'lrelu'
    CLRELU = 'clrelu'  # Clamped leaky relu.

    # Losses
    NLL_NORMAL = 'nll_normal'
    L1 = 'l1'
    MSE = 'mse'

    # Model output names
    OUT_MU = 'out_mu'
    OUT_SIGMA = 'out_sigma'
    OUT_RHO = 'out_rho'
    OUT_COEFFICIENT = 'out_coefficient'  # For GMM outputs only.
    OUT_BINARY = 'out_binary'  # For binary outputs and bernoulli loss

    # Suffix for output names.
    SUF_MU = '_mu'
    SUF_SIGMA = '_sigma'
    SUF_RHO = '_rho'
    SUF_COEFFICIENT = '_coefficient'  # For GMM outputs only.
    SUF_BINARY = '_binary'  # For binary outputs and bernoulli loss

    # Reduce function types
    R_MEAN_STEP = 'mean_step_loss'  # Take average of average step loss per sample over batch. Uses sequence length.
    R_MEAN_SEQUENCE = 'mean_sequence_loss' # Take average of sequence loss (summation of all steps) over batch. Uses sequence length.
    R_MEAN = 'mean' # Take mean of the whole tensor.
    R_SUM = 'sum'  # Take mean of the whole tensor.

    # Models
    MODEL_RNN = 'rnn'
    MODEL_BIRNN = 'birnn'  # Bidirectional RNN

    # Motion Datasets
    IMU = 'imu'

    # Dataset I/O keys for TF placeholders.
    PL_INPUT = "pl_input"
    PL_TARGET = "pl_target"
    PL_SEQ_LEN = "pl_seq_len"

    PL_ORI = "pl_imu_orientation"
    PL_ACC = "pl_imu_acceleration"
    PL_SMPL = "pl_imu_smpl_pose"

    # Preprocessing operations.
    PP_ZERO_MEAN_NORM = "pp_zero_mean_normalization"
    PP_IMU_ORI_NORM = "norm_ori"
    PP_IMU_ACC_NORM = "norm_acc"
    PP_IMU_SMPL_NORM = "norm_smpl"

    # Preprocessing operator side-effects.
    SE_PP_SEQ_LEN_DIFF = "sequence_length_diff"
    SE_PP_SEQ_LEN = "sequence_length"
    SE_PP_INPUT_SIZE = "input_feature_size"
    SE_PP_TARGET_SIZE = "target_feature_size"

    # Constants for demo
    SIZE_ORI = 45
    SIZE_ACC = 15
    SIZE_SMPL = 135

