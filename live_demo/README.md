# DIP: Live Demo
Code for running the live demo.

# Prerequisites
We used this code and the following software configuration on Windows 10.

## Python
Set up your environment as described for the training and evaluation environment (cf its [README](../train_and_eval/README.md)). Also make sure that the Python files in [train_and_eval/](../train_and_eval) are on your Python path and the pre-trained models are available.

## Unity
Download and install Unity from their official website. We used Unity 2017.3.1f1.

## SMPL
Once Unity is installed, download the Unity plugin for SMPL provided on their [website](http://smpl.is.tue.mpg.de/). Follow the provided instructions to enable SMPL in Unity.

## Xsens SDK
We used Xsens' MTw Awinda Wireless Motion Trackers to record IMU data. We interface with the Xsens SDK directly from within Unity, i.e. using C#. To use the scripts in this folder, you have to install [the software suite](https://www.xsens.com/mt-software-suite-mtw-awinda/). We used version 4.6.

To troubleshoot communication with the SDK, try the following:
- Make sure you can access the sensors by testing them with the MT Manager software, which is included in the software suite.
- Connecting Xsens with Unity is explained in one of their [tutorials](https://youtu.be/nXVzO8s4-zU). To set this up, you have to download a Unity plugin provided by Xsens, which contains some useful example scripts.
- The SDK contains some code examples written in C#, which might help you further.

## JsonDotNet
We use Json to send data between Unity and Python and make use of the [JsonDotNet](https://assetstore.unity.com/packages/tools/input-management/json-net-for-unity-11347) plugin.

## Setting up the Unity Project
Because of license considerations, we unfortunately cannot share the entire Unity project. However, here is a step-by-step explanation how to set up the project.

- Download Unity and create a new project.
- Download and install all the dependencies listed above (SMPL plugin, Xsens SDK, JsonDotNet).
- In the Asset folder, create a new folder, e.g. named `XSensStream`.
- Copy the following files into this folder:
  - The entire folder `%MT_SOFTWARE_SUITE%/MT SDK/Examples/awindamonitor_csharp/wrap_csharp64` where `%MT_SOFTWARE_SUITE%` is where you installed the Xsens Software Suite.
  - The files `MyXda.cs` and `MyEventArgs.cs` located in `%MT_SOFTWARE_SUITE%/MT SDK/Examples/awindamonitor_csharp`
  - All the C# scripts provided in this repository. Their functionality is briefly described in the following.

## Scripts Overview
**read_mtw.cs**
This is the main script that drives all the functionality of the system. It connects to the Xsens sensors, the Python server that returns the model's results (also called *inference server*) and performs the calibration.

**Client.cs**
Simple TCP client to connect to the inference server. It sends IMU data and receives the model's predictions.

**SMPLPoseUpdater.cs**
Helper class to animate the SMPL model. Contains some other useful functions to draw IMU data.

**IMUVisualizer.cs**
Helper class to store incoming Data from the IMU sensors.

**visualize_data.cs**
A script that can be used to visualize results that are stored on disk. It sends a filename to the inference server, which then loads its contents and streams them to Unity for visualization.

**inference_server.py**
A TCP server written in Python. It primarily loads the pre-trained models into a TensorFlow session and can then be queried to return the SMPL pose parameters for a given set of IMU readings. It can also load previously computed results stored as `.pkl` files and stream them to Unity for visualization.

# Example Usage
You can use the above mentioned scripts (i) to visualize results that were produced using the [evaluation code](../train_and_eval) or (ii) to run the live-demo as shown in the paper. Both modes require that the `inference_server.py` is running.

In the following we assume at least one SMPL model was dragged into the Unity scene to become a component. Please follow the rules explained in the SMPL Unity plugin when performaing this. This is important as otherwise vertex IDs will change which can result in unexpected behaviour. When the SMPL model is available as a component, you should see a child called `f_avg` or `m_avg` depending on whether you imported the female or male model. In the following, we refer to this component simply as the SMPL component.

## Visualization
To use the visualization code, add the script `visualize_data.cs` to the SMPL component. You should see the fields `OURS`, `SIP`, `SOP`, and `GT`. This script was used to produce the figures and videos in the paper, this is why it expects three other SMPL components to be available in the scene. You can create these components or just change the script to use only one.

These four fields are meant to contain names of the files that you want to load (e.g. a `.npz` file created by the evaluation code). The remaining three fields (`SIP Transform`, `SOP Transform`, `GT Transform`) should be mapped to the SMPL component of the three other models (in case you chose to create these as well). E.g., if you created another SMPL component in the scene to show a second motion, just drag it (the `m_avg` or `f_avg`) into one of these fields.

## Live Demo
To run the live demo, simply drag the script `read_mtw.cs` onto an SMPL component in your scene. We recommend to use radio channel 11 at 60 Hz. Also make sure the `Use Head Sensor as Input` option is toggled. Finally, you will have to change the expected sensor IDs at the top of `read_mtw.cs`. You can read out the IDs of your sensors using MT Manager provided in the Xsens software suite.

Before starting the live demo, don't forget to
- run the inference server. Check `inference_server.py` for some configurable parameters (e.g. which model to use, how many past and future frames, IP address, etc.).
- make sure the Awinda wireless station or Bluetooth dongle is connected and working properly.

When you start the live demo, a number of buttons are shown on the top left of the window. The workflow of the script is as follows:
- When running the Unity project, it will connect to the inference server and immediately prepare and load the TensorFlow model. Upon success, a corresponding message is printed both to the Unity and Python console.
- It also tries to connect to the Awinda station or Bluetooth dongle. If no connection can be established, a corresponding error message is printed to the Unity console as well.
- If connections to both Xsens and Python were successful, you can activate the 6 sensors whose IDs you defined in `read_mtw.cs`. Wait until they all blink in unison with the Awinda wireless station or dongle. The Unity console will also tell you that sensors have connected.
- Once all 6 sensors have connected, you are ready enter to measuring mode, i.e. start streaming data from the sensors. To do so, hit the `Start Measurement` button.
- The SMPL model should now move - weirdly, because there is no calibration yet. Instruct the subject to perform the calibration pose as explained in the paper. Then hit the `Calibrate` button. The `Calibrate` button is only available if the head sensor is levelled, i.e. pitch and roll are within a reasonable range around 0. We implemented this to ensure a certain quality of the calibration. If you don't like this, you can disable it by setting `_calibrationEnabled` in `read_mtw.cs` to true always.
- After the calibration, you might want to check that it was successful by telling the subject to do some simple movements.
- Now the system is ready. Hit `Toggle Model Inference` to get the predicted poses from the inference server and enjoy the full output of DIP. Hit it again to disable inference.

The remaining buttons are mainly for debugging purposes. To shut down the live demo, just stop the Unity project, which will stop communications gracefully.

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