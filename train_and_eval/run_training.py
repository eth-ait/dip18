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
from tf_train import TrainingEngine
from constants import Constants
from configuration import Configuration

import argparse
import os
import glob

C = Constants()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    Configuration.define_training_setup(parser)

    args = parser.parse_args()
    args_dict = vars(args)

    if args.model_id is not None:
        # Restore
        if len(args.model_id) == 10:
            model_dir = glob.glob(os.path.join(args.save_dir, "tf-" + args.model_id + "-*"), recursive=False)[0]
        else:
            model_dir = os.path.join(args.save_dir, args.model_id)
        config_dict = Configuration.from_json(os.path.abspath(os.path.join(model_dir, 'config.json')))

        # In case the experiment folder is renamed, update configuration.
        config_dict['model_dir'] = model_dir
        # 'model_type' parameter in args_dict should be the same as the one loaded from the json
        args_dict['model_type'] = config_dict['model_type']

        config = Configuration(**{**config_dict, **args_dict})
    else:
        # Start a new experiment from a json file.
        if args.json_file is not None:
            config_dict = Configuration.from_json(args.json_file)
        else:
            config_dict = Configuration.from_template(args.data_type, args.model_type)

        config_dict['model_dir'] = None
        override_args = ['save_dir', 'system', 'data_file', 'json_file']

        for a in override_args:
            config_dict[a] = getattr(args, a)
        config = Configuration(**config_dict)

    is_fine_tuning = False
    experiment_name = ""
    data_stats = None
    if hasattr(args, 'finetune_train_data') and args.finetune_train_data is not None and args.finetune_valid_data is not None:
        config.override_data_path(args.finetune_train_data, args.finetune_valid_data)
        config.set('evaluate_every_step', 50, override=True)
        config.set('print_every_step', 50, override=True)
        config.set('learning_rate', 1e-4, override=True)
        is_fine_tuning = True
        experiment_name = "fine_tuning"
    elif hasattr(args, 'finetune_train_data') and args.finetune_train_data is not None:
        config.override_data_path(args.finetune_train_data)
        is_fine_tuning = True
        config.set('evaluate_every_step', 50, override=True)
        config.set('print_every_step', 50, override=True)
        config.set('learning_rate', 1e-4, override=True)
        experiment_name = "fine_tuning"

    print('system', config.config['system'])
    config.set_experiment_name(experiment_name=experiment_name)
    tf.set_random_seed(config.get('seed'))
    training_engine = TrainingEngine(config, args.analyze_after_training,  early_stopping_tolerance=10,
                                     is_fine_tuning=is_fine_tuning, data_stats=data_stats)
    training_engine.run()
