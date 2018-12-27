# DIP: Training and Evaluation
Code for training of new and evaluation of existing models. 

# Prerequisites
It requires that you download the data from [(coming soon)](#).

The code was tested with TensorFlow 1.4.0, but it should be able to run in newer versions as well. Set up your environment as follows (assuming that correct CUDA and cuDNN versions are installed):

```commandline
conda create -n tf_py35 python=3.5
source activate tf_py35
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
conda install opencv
pip install numpy-quaternion
conda install numba
```

Next, make sure to update the path to where you stored the data (`*.npz` files) under `Configuration.KNOWN_DATA_PATHS` in `configuration.py`. The keys in that dictionary can be selected through the `--system` flag, so that you can run the code on different machines.

# Pre-Trained Models
With this code, we also release some pre-trained models on the [project page](http://dip.is.tue.mpg.de/pre_download). Unzip the downloaded file and store it under `./models` so that the evaluation code can make use of it. The zip file contains the following models:

- ID `1527876409`: The best BiRNN as reported in the paper (using dropout and acceleration loss)
- ID `1528208085`: The best BiRNN fine-tuned on DIP-IMU.

# Example Usage
## Training From Scratch
To re-train the best BiRNN as reported in the paper, use the following command line:

```commandline
python run_training.py --save_dir ./models --system local --data_file v9 --json ./models/tf-1527876409-imu_v9-birnn-fc1_512-lstm2_512-idrop2-relu-norm_ori_acc_smpl-auxloss_acc/config.json
```

This loads the configuration of the BiRNN from a json file and saves the checkpoints into `./models` (a new directory will be created). `--data_file v9` chooses the synthetic data to train on.

## Fine-tuning
To fine-tune a model on DIP-IMU use the following command.

```commandline
python run_training.py --save_dir ./models --model_id 1527876409 --system local --data_file v9 --norm_ori --norm_acc --norm_smpl --use_acc_loss --finetune_train_data imu_own_training.npz --finetune_valid_data imu_own_test.npz
```

This will load the model with the given ID (must be available under `./models/tf-1527876409-...`) and fine-tunes on DIP-IMU (a file `imu_own_training.npz` must be available in the data directory). `--norm_[ori,acc,smpl]` specifies that the input orientations and accelerations as well as the SMPL targets are normalized to have zero-mean unit-variance. `--use_acc_loss` enforces reconstruction of the accelerations in the output as specified in the paper. `--finetune_valid_data` is optional but let's you observe the performance on a held out data set directly in tensorboard. Also, early stopping is applied based on the performance on `--finetune_valid_data`.

## Evaluation (offline)
To evaluate the best BiRNN on the test set of DIP-IMU, use the following command.

```commandline
python run_evaluation.py --system local --data_file v9 --model_id 1527876409 --save_dir ./models
--eval_dir ./evaluation_results/ --datasets dip-imu
```

This will load the model with the given ID (must be available under `./models/tf-1527876409-...`) and print metrics computed over DIP-IMU (`dip-imu`). The results are also dumped to a log file in `--eval_dir`. If you want to visualize some results, add `--save_predictions` to the command. This will dump all the samples in the respective database as a `.npz` file. You can visualize these files using the code available in the live demo [folder](../live_demo).

## Evaluation (online)
To evaluate a BiRNN using only windowed input, use the following command.

```commandline
python run_evaluation.py --system local --data_file v9 --model_id 1527876409 --save_dir ./models
--eval_dir ./evaluation_results/ --datasets dip-imu --past_frames 20 50 --future_frames 5
```

This will evaluate the model in online mode on DIP-IMU, once using 20 past and 5 future frames and once using 50 past and 5 future frames.
 
# Contact Information
For questions or problems please file an issue or contact [manuel.kaufmann@inf.ethz.ch](mailto:manuel.kaufmann@inf.ethz.ch) or [yinghao.huang@tuebingen.mpg.de](mailto:yinghao.huang@tuebingen.mpg.de).

# Citation
If you use this code or data for your own work, please use the following citation:

```commandline
@article{DIP:SIGGRAPHAsia:2018,
	title = {Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time},
    	author = {Huang, Yinghao and Kaufmann, Manuel and Aksan, Emre and Black, Michael J. and Hilliges, Otmar and Pons-Moll, Gerard},
    	journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    	volume = {37},
    	pages = {185:1-185:15},
    	publisher = {ACM},
    	month = nov,
    	year = {2018},
    	note = {First two authors contributed equally},
    	month_numeric = {11}
}
```
