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
import argparse
import glob

import utils
from tf_models import *
from tf_data_feeder import DataFeederTF
from configuration import Configuration
from utils import Logger


def undo_smpl(dataset_obj, pose, mask=None, aa_representation=False):
    """
    Undo normalization on SMPL targets and add joints that are not predicted by the model.
    """
    smpl_samples = pose
    if mask is not None:
        smpl_samples = utils.padded_array_to_list(smpl_samples, mask)

    smpl_samples = dataset_obj.undo_preprocess_smpl(smpl_samples)
    smpl_samples = [utils.smpl_reduced_to_full(p) for p in smpl_samples]

    if aa_representation:
        smpl_samples = utils.rot_to_aa_representation(smpl_samples)

    return smpl_samples


def do_evaluation(config, datasets, len_past, len_future, save_predictions=False, verbose=0):
    """
    Evaluate the given model on all given datasets.
    :param config: Config to create model.
    :param datasets: List of tuples specifying name and batch size per dataset.
    :param len_past: Number of past frames to use (BiRNN only).
    :param len_future: Number of future frames to use (BiRNN only).
    :param save_predictions: Whether or not to save predictions as pkl files.
    :param verbose: Verbosity level.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    birnn_eval_chunks = False
    model_stamp = config.get('model_id').split("-")[1]
    eval_str = ""
    if config.get("model_type") == C.MODEL_BIRNN:
        if len_past >= 0:
            print("\nBiRNN is evaluated on chunks: " + str(len_past) + "_" + str(len_future))
            birnn_eval_chunks = True
        else:
            print("\nBiRNN is evaluated on the whole sequence.")

        if birnn_eval_chunks:
            eval_str = "past_{}_future_{}_frames".format(len_past, len_future)
            model_stamp += "_p{}_f{}".format(len_past, len_future)
        else:
            eval_str = "all_frames"
            model_stamp += "_all"

    model_cls = config.model_cls
    dataset_cls = config.dataset_cls

    # Data preprocessing configuration.
    preprocessing_ops = config.get_preprocessing_ops()

    # Logger object.
    logger = Logger(os.path.join(config.get('eval_dir'), "evaluation.txt"), sys.stdout)
    performance_text_format = "*** {} (SIP error): {:.4f} (+/- {:.3f})\n"
    performance_text_over_datasets = "\nSummary of model " + config.get('model_id') + "\n"

    for eval_key, batch_size in datasets:
        logger.print('------------------------------------------')
        logger.print('\nEvaluation on ' + eval_key)
        logger.print('\n------------------------------------------\n')

        # Clean slate.
        tf.reset_default_graph()

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            queue_threads = []

            prediction_list = []
            gt_list = []
            if config.get(eval_key, None) is None:
                print("Eval Key {} not found, continue.".format(eval_key))
                continue

            eval_dataset = dataset_cls(config.get(eval_key),
                                       var_len_seq=True,
                                       preprocessing_ops=preprocessing_ops)
            assert eval_dataset.num_samples % batch_size == 0, 'number of samples must be divisible by batch size'
            num_eval_iterations = int(eval_dataset.num_samples/batch_size)

            with tf.name_scope(eval_key):
                eval_data_feeder = DataFeederTF(eval_dataset, 1, batch_size, queue_capacity=1024, shuffle=False)
                data_placeholders = eval_data_feeder.batch_queue(dynamic_pad=eval_dataset.is_dynamic,
                                                                 queue_capacity=512,
                                                                 queue_threads=2)
                eval_model = model_cls(config=config,
                                       session=sess,
                                       reuse=False,
                                       mode="validation",
                                       placeholders=data_placeholders,
                                       input_dims=eval_dataset.input_dims,
                                       target_dims=eval_dataset.target_dims,
                                       data_stats=None)
                eval_model.build_graph()

                # Load variables
                try:
                    saver = tf.train.Saver()
                    # Restore variables.
                    if config.get('checkpoint_id') is None:
                        checkpoint_path = tf.train.latest_checkpoint(config.get("model_dir"))
                    else:
                        checkpoint_path = os.path.join(config.get("model_dir"), config.get("checkpoint_id"))

                    print("Loading model " + checkpoint_path)
                    saver.restore(sess, checkpoint_path)
                except Exception:
                    raise Exception("Could not load variables.")

            # In case we want to use feed dictionary.
            tf_mask = tf.expand_dims(tf.sequence_mask(lengths=data_placeholders[C.PL_SEQ_LEN], dtype=tf.float32), -1)
            tf_data_fetch = dict()
            tf_data_fetch['targets'] = data_placeholders[C.PL_TARGET]
            tf_data_fetch['mask'] = tf_mask
            tf_data_fetch['inputs'] = data_placeholders[C.PL_INPUT]

            eval_data_feeder.init(sess, coord)
            queue_threads.extend(tf.train.start_queue_runners(coord=coord, sess=sess))
            queue_threads.append(eval_data_feeder.enqueue_threads)

            total_loss = 0.0
            total_loss_l2 = 0.0
            n_data = 0
            dof = 9

            # where the sensors are attached
            tracking_sensors = [4, 5, 18, 19, 0, 15]
            sip_eval_sensors = [1, 2, 16, 17]

            # the remaining "sensors" are evaluation sensors
            all_sensors = utils.SMPL_MAJOR_JOINTS if config.get('use_reduced_smpl') else list(
                range(utils.SMPL_NR_JOINTS))
            remaining_eval_sensors = [s for s in all_sensors if s not in tracking_sensors and s not in sip_eval_sensors]

            with utils.Stats(tracking_sensors, sip_eval_sensors, remaining_eval_sensors, logger) as stats:
                model_evaluation_ops = dict()
                model_evaluation_ops['loss'] = eval_model.ops_loss
                model_evaluation_ops['mask'] = eval_model.seq_loss_mask
                model_evaluation_ops['targets'] = eval_model.pl_targets
                model_evaluation_ops['prediction'] = eval_model.output_sample
                model_evaluation_ops['orientation'] = eval_model.orientation
                model_evaluation_ops['acceleration'] = eval_model.acceleration

                for i in range(num_eval_iterations):
                    if verbose > 0 and ((i+1) % max(int((num_eval_iterations/5)), 1) == 0):
                        print(str(i+1) + "/" + str(num_eval_iterations))

                    if birnn_eval_chunks:
                        np_batch = sess.run(tf_data_fetch)
                        eval_out = eval_model.model.reconstruct_chunks(input_sequence=np_batch['inputs'],
                                                                       target_sequence=np_batch['targets'],
                                                                       len_past=len_past,
                                                                       len_future=len_future)
                        eval_out['mask'] = np_batch['mask']
                        eval_out['targets'] = np_batch['targets']
                        eval_out['prediction'] = eval_out['sample']
                    else:
                        eval_out = sess.run(model_evaluation_ops, feed_dict={})

                    total_loss += eval_out['loss']['total_loss']*batch_size
                    n_data += batch_size

                    pred = undo_smpl(eval_dataset, eval_out['prediction'], eval_out['mask'][:, :, 0])
                    targ = undo_smpl(eval_dataset, eval_out['targets'], eval_out['mask'][:, :, 0])

                    if save_predictions:
                        prediction_list.extend(pred)
                        gt_list.extend(targ)

                    # replace root with sensor data
                    for j in range(batch_size):
                        imu_root = np.reshape(np.eye(3), [-1]) if dof == 9 else np.array([1.0, 0.0, 0.0, 0.0])
                        pred[j][:, :dof] = imu_root
                        targ[j][:, :dof] = imu_root

                        ja_diffs, euc_diffs = utils.compute_metrics(prediction=pred[j:j + 1],
                                                                    target=targ[j:j + 1],
                                                                    compute_positional_error=False)
                        stats.add(ja_diffs, euc_diffs)

                total_loss = total_loss/float(n_data) if n_data > 0 else 0.0
                total_loss_l2 = total_loss_l2/float(n_data) if n_data > 0 else 0.0

                logger.print('\n*** Loss ***\n')
                logger.print('average main loss per time step: {}\n'.format(total_loss))
                logger.print('average l2 loss per time step  : {}\n'.format(total_loss_l2))
                sip_stats = stats.get_sip_stats()

            performance_text_over_datasets += performance_text_format.format(eval_key, sip_stats[0], sip_stats[1])

            if save_predictions:
                out = {"prediction": prediction_list,
                       "gt": gt_list}
                file_name = eval_key + "_" + eval_str if eval_str is not None else eval_key
                np.savez_compressed(os.path.join(config.get("eval_dir"), file_name), **out)

            sess.run(eval_data_feeder.input_queue.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(queue_threads, ignore_live_threads=True, stop_grace_period_secs=1)

    logger.print(performance_text_over_datasets)
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.define_evaluation_setup(parser)
    args = parser.parse_args()

    try:
        model_id = int(args.model_id)
    except ValueError:
        raise Exception("model_id is expected to be an integer")

    model_dir = glob.glob(os.path.join(args.save_dir, "tf-" + args.model_id + "-*"), recursive=False)[0]

    # Create configuration object.
    config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, 'config.json')))

    # Some arguments override the stored config parameters.
    config_dict['system'] = args.system
    config_dict['data_file'] = args.data_file if args.data_file is not None else config_dict['data_file']
    config_obj = Configuration(**config_dict)
    config_obj.set('checkpoint_id', args.checkpoint_id, override=True)

    if args.eval_dir is None:
        config_obj.set('eval_dir', os.path.join(model_dir, "evaluation"), override=True)
    else:
        config_obj.set('eval_dir', os.path.join(args.eval_dir, config_obj.get('model_id')), override=True)
    config_obj.set('model_dir', model_dir, override=True)  # in case folder is renamed.

    # Create evaluation directory if it does not exist yet
    if not os.path.exists(config_obj.get('eval_dir')):
        os.makedirs(config_obj.get('eval_dir'))

    # Past frames and future frames only valid for BiRNN.
    past_frames = args.past_frames
    future_frames = args.future_frames
    if config_obj.get("model_type") != C.MODEL_BIRNN:
        print("\nEvaluation with windows is only valid for BiRNN models.")
        past_frames = [-1]
        future_frames = [-1]

    # Parse which datasets to evaluate (name and batch_size).
    known_datasets = {'dip-imu': ('test_our_data', 18),
                      'tc': ('test_total_capture', 45),
                      'playground': ('test_playground_data', 7)}
    to_evaluate = [known_datasets[e.lower()] for e in args.datasets]

    for len_past_ in past_frames:
        for len_future_ in future_frames:
            do_evaluation(config=config_obj,
                          datasets=to_evaluate,
                          len_past=len_past_,
                          len_future=len_future_,
                          verbose=args.verbose,
                          save_predictions=args.save_predictions)
