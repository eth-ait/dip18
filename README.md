# Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time

## Code
This repository contains the code published alongside with our SIGGRAPH Asia [paper](http://dip.is.tuebingen.mpg.de/assets/dip.pdf). It is organised as follows: [`train_and_eval`](train_and_eval) contains code to train and evaluate the neural networks proposed in the paper. [`live_demo`](live_demo) contains Unity and Python scripts to use the models for real-time inference. [`data_synthesis`](data_synthesis) contains a script to produce synthetic IMU measurements from SMPL sequences. Please refer to the READMEs in the respective subfolder for more details.

## Data
To download the data please visit the [project page](http://dip.is.tuebingen.mpg.de). From this page you can also download the SMPL reference parameters for the TotalCapture dataset. To preprocess TotalCapture data, please refer to [`read_TC_data.py`](data_synthesis/read_TC_data.py).

## Visualization
Apart from the live demo, this repository does not offer any other visualization tools. However, the data can easily be visualized with the [aitviewer](https://github.com/eth-ait/aitviewer). The examples provided by aitviewer contain two scripts that load data associated with DIP:
- [Loading ground-truth SMPL poses and IMUs from the DIP-IMU dataset.](https://github.com/eth-ait/aitviewer/blob/main/examples/load_DIP_IMU.py)
- [Loading ground-truth SMPL poses and IMUs from the TotalCapture dataset.](https://github.com/eth-ait/aitviewer/blob/main/examples/load_DIP_TC.py)

![DIP-IMU rendering with aitviewer](DIP_IMU_example.gif)

## Contact Information
For questions or problems please file an issue or contact [manuel.kaufmann@inf.ethz.ch](mailto:manuel.kaufmann@inf.ethz.ch) or [yinghao.huang@tuebingen.mpg.de](mailto:yinghao.huang@tuebingen.mpg.de).

## Citation
If you use this code or data for your own work, please cite:

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
