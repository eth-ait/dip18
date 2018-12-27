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

A training script that can be used for basic tasks.

- Loads model and dataset classes given by config.
- Creates dataset and data feeder objects for training.
- Creates training model.
- If validation data is provided, creates validation data & data feeder and validation model. Note that validation model
uses a different computational graph but shares its weights with the training model.
- Standard tensorflow routines (i.e., session creation, gradient checks, optimization, summaries, etc.).
- Main training loop:
    * Graph ops and summary ops to be evaluated are defined by the model class.
    * Model is evaluated on the full validation data every time. Because of tensorflow input queues, we use an
    unconventional routine. We need to iterate `num_validation_iterations` (# validation samples/batch size) times.
    Model keeps track of losses and report via `get_validation_summary` method.
"""
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import os
import numpy as np
from tf_data_feeder import DataFeederTF, TFStagingArea
from utils import get_model_dir_timestamp, create_tf_timeline


class TrainingEngine(object):
    def __init__(self, config, analyze_after_training=False, early_stopping_tolerance=15, is_fine_tuning=False, data_stats=None):
        self.config = config
        self.analyze_after_training = analyze_after_training
        self.model_dir = config.get('model_dir')
        self.tensorboard_verbosity = config.get('tensorboard_verbose')  # Define detail level of tensorboard plots.
        self.is_fine_tuning = is_fine_tuning

        # Training loop setup.
        self.training_num_epochs = config.get('num_epochs') + 1
        self.training_evaluate_every_step = config.get('evaluate_every_step')
        self.training_create_timeline = config.get('create_timeline', False)
        self.early_stopping_save = early_stopping_tolerance > 0
        self.early_stopping_tolerance = early_stopping_tolerance
        self.training_checkpoint_every_step = config.get(
            'checkpoint_every_step') if not self.early_stopping_save else self.training_evaluate_every_step

        # Data preprocessing configuration.
        self.preprocessing_ops = config.get_preprocessing_ops()

        # Create a session object and initialize parameters.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth=True)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        # Create step counter (used by optimization routine and schedulers.)

        # with tf.variable_scope("global_step"):
        self.global_step = tf.Variable(1, trainable=False, name='global_step')

        if config.get('learning_rate_type') == 'exponential':
            self.learning_rate = tf.train.exponential_decay(config.get('learning_rate'),
                                                            global_step=self.global_step,
                                                            decay_steps=config.get('learning_rate_decay_steps'),
                                                            decay_rate=config.get('learning_rate_decay_rate'),
                                                            staircase=False)
            tf.summary.scalar('training/learning_rate', self.learning_rate, collections=["training_status"])
        elif config.get('learning_rate_type') == 'fixed':
            self.learning_rate = config.get('learning_rate')
        else:
            raise Exception("Invalid learning rate type")

        self.model_cls = config.model_cls
        self.dataset_cls = config.dataset_cls

        # Training model
        self.training_dataset, self.num_training_iterations = self.load_dataset(config.get('training_data'), data_stats)
        print("# training steps per epoch: " + str(self.num_training_iterations))

        # Validation model
        self.apply_validation = config.get('validate_model', False)
        if self.apply_validation:
            self.validation_dataset, self.num_validation_iterations = self.load_dataset(config.get('validation_data'),
                                                                                        data_stats)
            assert not (self.num_validation_iterations == 0), "Not enough validation samples."
            print("# validation steps per epoch: " + str(self.num_validation_iterations))

    def run(self):
        # Models in different modes (training, validation, sampling, etc.)
        self.create_models()
        # Gradient clipping
        self.gradient_check()
        # Tensorflow routines
        self.call_tensorflow_routines()
        # Summary writer
        self.create_summaries()
        # Save configuration in pickle and json formats.
        self.config.dump(self.config.get('model_dir'))
        # Main training loop.
        self.train()
        # Close input queues and stop threads.
        self.finalize_training()

    def load_dataset(self, path, data_stats=None):
        dataset = self.dataset_cls(path, preprocessing_ops=self.preprocessing_ops, data_stats=data_stats)
        print("loaded {} ({} samples, batch size {})".format(path, dataset.num_samples, self.config.get('batch_size')))
        num_data_iterations = max(1, int(dataset.num_samples / self.config.get('batch_size')))
        return dataset, num_data_iterations

    def create_model_graph(self, dataset, mode, reuse):
        # Create a tensorflow sub-graph that loads batches of samples.
        # (1) Create input pipeline
        data_feeder = DataFeederTF(dataset, self.config.get('num_epochs'), self.config.get('batch_size'),
                                   queue_capacity=1024)
        data_placeholders = data_feeder.batch_queue(dynamic_pad=dataset.is_dynamic,
                                                    queue_capacity=512,
                                                    queue_threads=4)

        # (2) Create staging area for faster data transfer to GPU memory.
        if self.config.get('use_staging_area', False):
            staging_area = TFStagingArea(data_placeholders, device_name="/gpu:0")
            data_placeholders = staging_area.tensors
        else:
            staging_area = None

        # (3) Create model.
        with tf.name_scope(mode):
            model = self.model_cls(config=self.config,
                                   session=self.session,
                                   reuse=reuse,
                                   mode=mode,
                                   placeholders=data_placeholders,
                                   input_dims=dataset.input_dims,
                                   target_dims=dataset.target_dims,
                                   data_stats=None)
            model.build_graph()

        return model, data_feeder, staging_area

    def create_models(self):
        """
        Create trainind and validation models as separate computational graphs.
        """
        self.training_model, self.training_data_feeder, self.training_staging_area = self.create_model_graph(
            dataset=self.training_dataset, mode='training', reuse=False)

        # Preparing lists of objects to initialize/run later.
        self.data_feeders = [self.training_data_feeder]
        self.staging_areas = []
        if self.training_staging_area:
            self.staging_areas = [self.training_staging_area]
            self.training_model.register_run_ops('staging_area', self.training_staging_area.preload_op)

        if self.apply_validation:
            self.validation_model, self.validation_data_feeder, self.validation_staging_area = self.create_model_graph(
                dataset=self.validation_dataset, mode='validation', reuse=True)
            self.data_feeders.append(self.validation_data_feeder)
            if self.validation_staging_area:
                self.staging_areas.append(self.validation_staging_area)
                self.validation_model.register_run_ops('staging_area', self.validation_staging_area.preload_op)

    def create_summaries(self):
        """
        Registers and creates summary ops related with training status.
        """
        summary_dir = os.path.join(self.model_dir, "summary")
        self.summary_writer = tf.summary.FileWriter(summary_dir, self.session.graph)

        # Create summaries to visualize weights and gradients.
        if self.tensorboard_verbosity > 2:
            for grad, var in self.grads_and_vars:
                tf.summary.histogram(var.name, var, collections=["training_status"])
                tf.summary.histogram(var.name + '/gradient', grad, collections=["training_status"])

        # Create summary to visualize input queue load level.
        if self.tensorboard_verbosity > 0:
            tf.summary.scalar("training/queue",
                              math_ops.cast(self.training_data_feeder.input_queue.size(), dtypes.float32) * (
                                          1. / self.training_data_feeder.queue_capacity),
                              collections=["training_status"])

        self.training_summary = tf.summary.merge_all('training_status')
        self.training_model.register_run_ops('summary', self.training_summary)

    def gradient_check(self):
        """
        Applies gradient clipping and sets train_op.
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # Gradient clipping.
            grads = tf.gradients(self.training_model.loss, tf.trainable_variables())
            if self.config.get('grad_clip_by_norm') > 0:
                grads, global_norm = tf.clip_by_global_norm(grads, self.config.get('grad_clip_by_norm'))
                tf.summary.scalar('training/gradient_norm', global_norm, collections=["training_status"])

            self.grads_and_vars = list(zip(grads, tf.trainable_variables()))
            if self.config.get('grad_clip_by_value') > 0:
                grads_and_vars_clipped = []
                for grad, var in self.grads_and_vars:
                    grads_and_vars_clipped.append((tf.clip_by_value(grad, -self.config.get('grad_clip_by_value'),
                                                                    -self.config.get('grad_clip_by_value')), var))
                self.grads_and_vars = grads_and_vars_clipped

            self.train_op = optimizer.apply_gradients(grads_and_vars=self.grads_and_vars, global_step=self.global_step)
            self.training_model.register_run_ops('train_op', self.train_op)

    def call_tensorflow_routines(self):
        """
        Creates and runs basic tensorflow routines such as initialization, saver, coordinator, etc.
        """
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init_op)

        self.run_opts = None
        self.run_metadata = None
        if self.config.get('create_timeline', False):
            self.run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, timeout_in_ms=100000)
            self.run_metadata = tf.RunMetadata()

        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
        if self.model_dir:
            # If model directory already exists, continue training by restoring computation graph.
            # Restore variables.
            if self.config.get('checkpoint_id'):
                checkpoint_path = os.path.join(self.model_dir, self.config.get('checkpoint_id'))
            else:
                checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

            print("Continue training with model " + checkpoint_path)
            self.saver.restore(self.session, checkpoint_path)

            # Estimate training epoch.
            step = tf.train.global_step(self.session, self.global_step)
            self.start_epoch = round(step / (self.training_dataset.num_samples / self.config.get('batch_size')))

        else:
            # Fresh start
            # Create a unique output directory for this experiment.
            model_timestamp = get_model_dir_timestamp(prefix="tf", suffix=self.config.get('experiment_name'),
                                                      connector="-")
            self.model_id = model_timestamp
            self.config.set('model_id', model_timestamp, override=True)
            self.model_dir = os.path.abspath(os.path.join(self.config.get('save_dir'), model_timestamp))
            print("Saving to {}\n".format(self.model_dir))
            self.config.set('model_dir', self.model_dir, override=True)
            self.start_epoch = 1

        if self.is_fine_tuning:
            # Create a unique output directory for this experiment.
            model_timestamp = get_model_dir_timestamp(prefix="tf", suffix=self.config.get('experiment_name'),
                                                      connector="-")
            self.model_id = model_timestamp
            self.config.set('model_id', model_timestamp, override=True)
            self.model_dir = os.path.abspath(os.path.join(self.config.get('save_dir'), model_timestamp))
            print("Fine-tuning Model.")
            print("Saving to {}\n".format(self.model_dir))
            self.config.set('model_dir', self.model_dir, override=True)
            self.start_epoch = 1
            self.session.run(tf.assign(self.global_step, 1))

        # Initialize data loader threads.
        self.coordinator = tf.train.Coordinator()
        for data_feeder in self.data_feeders:
            data_feeder.init(self.session, self.coordinator)

        self.queue_threads = tf.train.start_queue_runners(sess=self.session, coord=self.coordinator)
        for data_feeder in self.data_feeders:
            self.queue_threads.append(data_feeder.enqueue_threads)

        if self.config.get('use_staging_area', False):
            for staging_area in self.staging_areas:
                # Fill staging area first.
                for i in range(256):
                    _ = self.session.run(staging_area.preload_op, feed_dict={})

    def train(self):
        step = 0
        best_validation_loss = np.inf
        validation_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False
        for epoch in range(self.start_epoch, self.training_num_epochs):
            if stop_signal:
                break
            for epoch_step in range(self.num_training_iterations):
                step = tf.train.global_step(self.session, self.global_step)

                run_training_output = self.training_model.training_step(step, epoch, feed_dict={})
                for summary_entry in run_training_output['summary']:
                    self.summary_writer.add_summary(summary_entry, step)

                if self.apply_validation and step % self.training_evaluate_every_step == 0:
                    validation_summary, validation_loss = self.validation_model.evaluation_step(step, epoch,
                                                                                                self.num_validation_iterations)
                    self.summary_writer.add_summary(validation_summary, step)

                    if validation_loss <= best_validation_loss:
                        num_steps_wo_improvement = 0
                    else:
                        num_steps_wo_improvement += 1

                    if num_steps_wo_improvement == self.early_stopping_tolerance:
                        stop_signal = True
                        break

                if self.training_create_timeline:
                    create_tf_timeline(self.model_dir, self.run_metadata)

                if (step % self.training_checkpoint_every_step) == 0 and validation_loss <= best_validation_loss:
                    ckpt_save_path = self.saver.save(self.session, os.path.join(self.model_dir, 'model'),
                                                     self.global_step - 1)
                    print("Model saved in file: %s" % ckpt_save_path)
                    best_validation_loss = min(best_validation_loss, validation_loss)

        print("End-of-Training.")
        if stop_signal is False and validation_loss < best_validation_loss:
            ckpt_save_path = self.saver.save(self.session, os.path.join(self.model_dir, 'model'), self.global_step)
            print("Model saved in file: %s" % ckpt_save_path)
            print('Model is trained for %d epochs, %d steps.' % (self.config.get('num_epochs'), step))

    def finalize_training(self):
        try:
            for data_feeder in self.data_feeders:
                self.session.run(data_feeder.input_queue.close(cancel_pending_enqueues=True))
            self.coordinator.request_stop()
            self.coordinator.join(self.queue_threads, stop_grace_period_secs=5)
        except:
            pass

        self.session.close()
        tf.reset_default_graph()
