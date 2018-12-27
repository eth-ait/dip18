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
import socket
import tensorflow as tf
import struct
import json
import numpy as np
import quaternion
import time
import pickle as pkl
import datetime

from threading import Thread
from configuration import Configuration
from constants import Constants
from utils import SMPL_NR_JOINTS, smpl_reduced_to_full


# TODO adjust these parameters if necessary
IP = '0.0.0.0'
PORT = 9999
MODEL_DIR = "../train_and_eval/models/tf-1528208085-fine_tuning-imu_v9-birnn-fc1_512-lstm2_512-idrop2-relu-norm_" \
            "ori_acc_smpl-auxloss_acc"
BUFFER_MODE = ['REPEAT', 'WAIT'][1]
RNG = np.random.RandomState(42)
RAND_POSE = RNG.rand(72, 1)[:, 0] * 0.2
LEN_PAST_FRAMES = 20
LEN_FUTURE_FRAMES = 5
RECORDING_DIR = "./recordings"
C = Constants()
SYSTEM = 'local'


class InferenceRunner(Thread):
    def __init__(self, client_socket, address):
        super(InferenceRunner, self).__init__()
        self.socket = client_socket
        self.address = address
        print('accepted connection from {}:{}'.format(address[0], address[1]))

        self.model = None
        self.running = True

        self.buffer = []
        self.buffer_size = -1
        self.input_buffer_size = None

        self.raw_sensor_data = []  # what is arriving from XSens in every frame
        self.model_input_output = []  # the input to the model and the corresponding output

        self.sample_sequence = None
        self.next_frame = 0

    def run(self):
        try:
            while self.running:
                request = self.recv_one_message()
                if request is None:
                    self.running = False
                    break
                self.handle_request(request)
        except Exception as e:
            print(e)

        print('client terminated connection')
        if self.model is not None:
            print('closing Tensorflow session')
            self.model.sess.close()

    def dump_recordings(self):
        # dumps the cached recordings and clears the cache afterwards
        folder_name = datetime.datetime.now().strftime('%m-%d-%H%M%S-%G')
        while os.path.exists(os.path.join(RECORDING_DIR, folder_name)):
            folder_name = datetime.datetime.now().strftime('%m-%d-%H%M%S-%G')
        folder_path = os.path.join(RECORDING_DIR, folder_name)
        os.makedirs(folder_path)

        # store the recorded inputs
        with open(os.path.join(folder_path, 'recording_raw_sensor_data.pkl'), 'wb') as f:
            pkl.dump(self.raw_sensor_data, f, protocol=2)  # make it compatible with py2

        # store the recorded model_outputs in Unity friendly format
        with open(os.path.join(folder_path, 'recording_model_io.pkl'), 'wb') as f:
            pkl.dump(self.model_input_output, f, protocol=2)  # make it compatible with py2

        # store the configuration
        config = {'buffer_mode': BUFFER_MODE,
                  'past_frames': LEN_PAST_FRAMES,
                  'future_frames': LEN_FUTURE_FRAMES}
        with open(os.path.join(folder_path, 'config.pkl'), 'wb') as f:
            pkl.dump(config, f, protocol=2)  # make it compatible with py2

        print("recordings dumped to {}".format(folder_path))

        self.raw_sensor_data = []
        self.model_input_output = []

    def send_one_message(self, data):
        length = len(data)
        self.socket.sendall(struct.pack('h', length))
        self.socket.sendall(data)

    def recv_one_message(self):
        lengthbuf = self.recvall(2)
        if lengthbuf is None:
            return None
        length, = struct.unpack('h', lengthbuf)
        msg = self.recvall(length)
        return msg.decode()

    def recvall(self, count):
        buf = b''
        while count:
            newbuf = self.socket.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def handle_request(self, request_msg):
        if request_msg == '':
            self.running = False

        elif request_msg == 'STARTUP':
            # client requests to load the model
            self.prepare_model()
            self.send_one_message(b'READY')
            print('model loaded successfully')

        elif request_msg == 'IMU':
            start = time.time()
            # the next message will be imu data
            imu_measurement = self.recv_one_message()
            imu = json.loads(imu_measurement)

            # store the model input
            self.raw_sensor_data.append(imu)

            # print("read data from network {}".format((time.time() - start)*1000.0))
            # get the prediction from the model
            pred = self.inference_step(imu)

            # send output back to client
            self.send_pose(pred)

        elif request_msg == 'SAMPLELOAD':
            # load a new sample, the path will be sent by the client
            filename = self.recv_one_message()
            self.load_sample_sequence(filename)

        elif request_msg == 'SAMPLENEXT':
            # get the next frame of the loaded sample and send it
            self.send_next_sample_frame()

        elif request_msg == 'SAMPLEPREV':
            # get the previous frame of the loaded sample and send it
            self.send_previous_sample_frame()

        elif request_msg == 'DUMP':
            # dump the recordings
            self.dump_recordings()

        elif request_msg == 'TEST':
            # just return a dummy string
            self.send_one_message(b"this but a scratch")

        else:
            print('unknown request {}'.format(request_msg))

    def prepare_model(self):
        """Prepare the configured model for inference."""
        # reset tensorflow computation graph
        tf.reset_default_graph()
        sess = tf.Session()
        self.model = load_model(sess, MODEL_DIR)
        self.buffer_size = LEN_PAST_FRAMES + 1 + LEN_FUTURE_FRAMES
        self.input_buffer_size = LEN_FUTURE_FRAMES + 1

    def inference_step(self, imu_data):
        """Returns the loaded model's prediction for the given set of IMU data."""
        # start = time.time()

        # if queue is empty, repeat the frame we received for as many times as required
        if BUFFER_MODE == 'REPEAT':
            if len(self.buffer) == 0:
                for _ in range(self.buffer_size - 1):
                    self.buffer.append(imu_data)

            # append newest frame to the right
            self.buffer.append(imu_data)

        elif BUFFER_MODE == 'WAIT':
            self.buffer.append(imu_data)

            if len(self.buffer) < self.input_buffer_size:
                # return a dummy value so that client knows it must wait until buffer is full
                print('waiting for buffer to fill up')
                return np.array([-1.0])

        else:
            raise RuntimeError('Buffer Mode "{}" unknown'.format(BUFFER_MODE))

        dynamic_buffer_size = len(self.buffer)
        # assert dynamic_buffer_size == self.input_buffer_size or dynamic_buffer_size <= self.buffer_size

        # assemble as many frames as we need from the buffer
        oris = []
        accs = []
        for i in range(dynamic_buffer_size):
            imu_m = self.buffer[i]
            oris.append(np.array(imu_m['orientations'])[np.newaxis, :])
            accs.append(np.array(imu_m['accelerations'])[np.newaxis, :])

        ori = np.concatenate(oris, axis=0)
        acc = np.concatenate(accs, axis=0)

        model_io = {'input_ori': np.copy(ori),
                    'input_acc': np.copy(acc)}

        # normalize w.r.t. root sensor
        root_idx = 5
        ori_n, acc_n = normalize(ori, acc, root_idx)

        # get the prediction from the model, previous state unused
        pred, _, x_input = self.model.inference_step(ori_n, acc_n, previous_state=None)  # (seq_length, dof)

        pred = pred[-(LEN_FUTURE_FRAMES+1):-LEN_FUTURE_FRAMES]
        ori = ori[-(LEN_FUTURE_FRAMES+1):-LEN_FUTURE_FRAMES]
        model_io['output_actual'] = np.copy(pred)

        root_ori = ori[:, 9 * root_idx:9 * (root_idx + 1)]

        # add unused joints
        prediction = smpl_reduced_to_full(pred)

        # replace root with root input sensor value
        prediction[:, 0:9] = root_ori

        # end = time.time()

        # convert to angle axis
        prediction = np.reshape(prediction, [SMPL_NR_JOINTS, 3, 3])
        prediction_aa = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(prediction))
        prediction_aa = np.reshape(prediction_aa, [-1])

        model_io['input_final'] = np.copy(x_input)
        model_io['output_final'] = np.copy(prediction_aa)
        self.model_input_output.append(model_io)

        # remove the first (oldest) frame from the buffer
        self.buffer = self.buffer[-self.buffer_size:]

        # print('\rinference done in {} ms'.format((end-start)*1000.0), end='')
        return prediction_aa

    def load_sample_sequence(self, filename):
        if filename.endswith('.pkl'):
            # This is a pickle file that contains a motion sequence stored in angle-axis format
            sample_sequence = pkl.load(open(filename, 'rb'))
            poses = np.array(sample_sequence)

        elif filename.endswith('.npz'):
            # This is a numpy file that was produced using the evaluation code
            preds = np.load(filename)['prediction']

            # Choose the first sample in the file to be displayed
            pred = preds[1]

            # This is in rotation matrix format, so convert to angle-axis
            poses = np.reshape(pred, [pred.shape[0], -1, 3, 3])
            poses = quaternion.as_rotation_vector(quaternion.from_rotation_matrix(poses))
            poses = np.reshape(poses, [poses.shape[0], -1])

        else:
            raise ValueError("do not recognize file format of file {}".format(filename))

        # Unity can also display orientations and accelerations, but they are not available here
        oris = np.zeros([poses.shape[0], 6 * 3])
        accs = np.zeros([poses.shape[0], 6 * 3])

        self.sample_sequence = {'gt': poses, 'ori': oris, 'acc': accs}
        self.next_frame = 0

    def send_next_sample_frame(self):
        # first send ground truth pose
        self.send_pose(self.sample_sequence['gt'][self.next_frame])

        # then send IMU data
        ori = self.sample_sequence['ori'][self.next_frame]
        acc = self.sample_sequence['acc'][self.next_frame]
        self.send_imu(ori, acc)

        self.next_frame += 1
        if self.next_frame >= len(self.sample_sequence['gt']):
            self.next_frame = 0

    def send_previous_sample_frame(self):
        prev_frame = self.next_frame - 2
        self.next_frame = self.next_frame - 1
        if prev_frame < 0:
            prev_frame = len(self.sample_sequence['gt']) - 1
            self.next_frame = 0

        self.send_pose(self.sample_sequence['gt'][prev_frame])
        ori = self.sample_sequence['ori'][prev_frame]
        acc = self.sample_sequence['acc'][prev_frame]
        self.send_imu(ori, acc)

    def send_pose(self, pose):
        """
        Send a 72 dimensional float array to the client.
        """
        print('\rsending frame {}'.format(self.next_frame), end='')
        pose_json = json.dumps({'pose': pose.tolist()}, separators=(',', ':'))
        self.send_one_message(pose_json.encode())

    def send_imu(self, orientations, accelerations):
        """
        Send a 54 dimensional float array (orientations) and 18 dimensional float array (accelerations) to the client.
        """
        imu_json = json.dumps({'orientations': orientations.tolist(),
                               'accelerations': accelerations.tolist()}, separators=(',', ':'))
        self.send_one_message(imu_json.encode())


def normalize(orientations, accelerations, root_idx, first_only=False):
    """
    Normalize orientations and accelerations with respect to `root_idx`. If `first_only` is true, then
    it will normalize only with respect to the root orientation occurring at the first frame in the sequence.
    `orientations` is expected in shape (seq_length, n*9) and `acceleration` in shape (seq_length, 6*3).
    The first five sensors are normalized and returned, i.e. two arrays of shape (seq_length, 5*9) and
    (seq_length, 5*3) respectively unless `first_only` is true in which case the first 6 sensors are returned.
    """
    seq_length = orientations.shape[0]
    oris = np.reshape(orientations, [seq_length, -1, 3, 3])
    accs = np.reshape(accelerations, [seq_length, -1, 3, 1])

    if first_only:
        first_root = oris[0:1, root_idx]
        first_root = np.repeat(first_root, seq_length, axis=0)
        root_inv = np.transpose(first_root, [0, 2, 1])
    else:
        root_inv = np.transpose(oris[:, root_idx], [0, 2, 1])

    # normalize orientations - if first_only is true we keep the root orientation, otherwise we discard it
    end_idx = 6 if first_only else 5
    oris_normalized = np.matmul(root_inv[:, np.newaxis, ...], oris)
    oris_normalized = np.reshape(oris_normalized[:, :end_idx], [seq_length, -1])

    # normalize accelerations
    accs_s = accs - accs[:, -1:]
    accs_s = accs_s[:, :end_idx]
    accs_normalized = np.matmul(root_inv[:, np.newaxis, ...], accs_s)
    accs_normalized = np.reshape(accs_normalized, [seq_length, -1])

    return oris_normalized, accs_normalized


def load_model(sess, model_dir):
    """Load the specified model into the given Tensorflow session."""
    config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, "config.json")))

    config_dict["system"] = SYSTEM
    config = Configuration(**config_dict)
    config.set('eval_dir', os.path.join(model_dir, "evaluation"), override=True)
    config.set('model_dir', model_dir, override=True)

    if not os.path.exists(config.get('eval_dir')):
        os.makedirs(config.get('eval_dir'))

    data_placeholders = dict()
    data_placeholders[C.PL_INPUT] = tf.placeholder(tf.float32, shape=[1, None, C.SIZE_ORI+C.SIZE_ACC])
    data_placeholders[C.PL_TARGET] = tf.placeholder(tf.float32, shape=[1, None, C.SIZE_SMPL])
    data_placeholders[C.PL_SEQ_LEN] = tf.placeholder(tf.int32, shape=[1])

    # Load data statistics
    stat_path = os.path.join(config.get('model_dir'), "stats.npz")
    assert os.path.exists(stat_path), "Data statistics file is required."
    stats = dict(np.load(stat_path))

    # Create Model
    model_cls = config.model_cls
    model = model_cls(config=config,
                      session=sess,
                      reuse=False,
                      mode="sampling",
                      placeholders=data_placeholders,
                      input_dims=[C.SIZE_ORI + C.SIZE_ACC],
                      target_dims=[C.SIZE_SMPL],
                      data_stats=stats)
    model.build_graph()

    # Restore weights
    try:
        saver = tf.train.Saver()
        checkpoint_path = tf.train.latest_checkpoint(config.get('model_dir'))

        print("Loading model " + checkpoint_path)
        saver.restore(sess, checkpoint_path)
    except Exception:
        raise Exception("Could not load model weights.")

    return model


if __name__ == '__main__':
    bind_ip = IP
    bind_port = PORT

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((bind_ip, bind_port))
    server.listen(1)

    print('Listening on {}:{}'.format(bind_ip, bind_port))

    while True:
        client_sock, address = server.accept()
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ir = InferenceRunner(client_sock, address)
        ir.start()


