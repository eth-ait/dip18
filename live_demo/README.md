# DIP: Live Demo
Code for running the live demo. Many thanks to Harun Salman for testing and providing extensive feedback to help improve the setup instructions.

# Setup Instructions
## OS Version
We used Windows 10 but it should also work with Windows 7 and potentially can run on macOS as well (untested).

## Python
Set up your environment as described for the training and evaluation environment (cf its [README](../train_and_eval/README.md)). Also make sure that the Python files in [train_and_eval/](../train_and_eval) are on your Python path and the pre-trained models are available.

## Unity
Download and install Unity from their official website. We used Unity 2017.3.1f1.

## SMPL
Once Unity is installed, download the Unity plugin for SMPL provided on their [website](http://smpl.is.tue.mpg.de/). Follow the provided instructions to enable SMPL in Unity.

## Xsens SDK
We used Xsens' MTw Awinda Wireless Motion Trackers to record IMU data. We interface with the Xsens SDK directly from within Unity, i.e. using C#. To use the scripts in this folder, you have to install [the software suite](https://www.xsens.com/mt-software-suite-mtw-awinda/). We used version 4.6.

If you experience problems when communicating with the SDK, try the following:
- Make sure you can access the sensors by testing them with the MT Manager software, which is included in the software suite.
- Connecting Xsens with Unity is explained in one of their [tutorials](https://youtu.be/nXVzO8s4-zU). To set this up, you have to download a Unity plugin provided by Xsens, which contains some useful example scripts.
- The SDK contains some code examples written in C#, which might help you further.

## JsonDotNet
We use Json to send data between Unity and Python and make use of the [JsonDotNet](https://assetstore.unity.com/packages/tools/input-management/json-net-for-unity-11347) plugin.

## Setting up the Unity Project
Because of license considerations we cannot share the entire Unity project. However, here is a step-by-step explanation how to set up the project.

- Download Unity and create a new project.
- Navigate to the Asset Store and install JsonDotNet for Unity (see above) if you haven't done so already.
- Download the SMPL plugin from the SMPL website (see above) and follow their installation instructions if you haven't done so already.
- Download the Xsens SDK (see above) if you haven't done so already.
- In the Asset folder, create a new folder, e.g. named `XSensStream`.
- Copy the following files into this folder:
  - The entire folder `%MT_SOFTWARE_SUITE%/MT SDK/Examples/awindamonitor_csharp/wrap_csharp64` where `%MT_SOFTWARE_SUITE%` is where you installed the Xsens Software Suite.
  - The files `MyXda.cs`, `MyEventArgs.cs`, `MasterInfo.cs`, and `DeviceInfo.cs` located in `%MT_SOFTWARE_SUITE%/MT SDK/Examples/awindamonitor_csharp`
  - All the C# scripts provided in this repository. Their functionality is briefly described below.
- Make sure the following configuration parameters are set appropriately (by default they should be set correctly):
  - In `inference_server.py` let the server listen to all incoming TCP/IP connections by setting `IP` to `0.0.0.0`. Set `PORT` to some number above 8000, e.g. `9999`.
  - Make sure the path under `MODEL_DIR` is correct. Under `MODEL_DIR` the models that you downloaded from the project page should be available.
  - Make sure `bindIP` and `bindPort` in `Client.cs` have been set accordingly, i.e., `127.0.0.0` and `9999`.
- As you are using a different set of Xsens sensors, you have to change the expected sensor IDs at the top of `read_mtw.cs`. You can read out the IDs of your sensors using MT Manager provided in the Xsens software suite. They are probably given as hex numbers as soon as the sensor connects to a master station. Convert them to decimal integers and write them onto lines 46-51 in `read_mtw.cs`. Make sure the mapping from sensor ID to actual location of the sensor on the body is correct!
- Finally, you need an SMPL avatar in the scene:
  - From the SMPL asset folder (`Assets/SMPL/Models`) import the female or male model *according to the specifications given in the SMPL tutorial*! This is important as otherwise vertex IDs might not correspond to the expected values.
  - On the imported model select the child `m_avg` (`f_avg` if you used the female model). Then, in the inspector tab use the `Add Component` button to add the `read_mtw.cs` file. This prepares the scene for the live demo usage. It is also possible to use the Unity project to visualize pre-recorded motions. More details to do so are given below.

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
A TCP server written in Python. It primarily loads the pre-trained models into a TensorFlow session and can then be queried to return the SMPL pose parameters for a given set of IMU readings. It can also load previously computed results stored as `.npz` files and stream them to Unity for visualization. It should be straight-forward to extend this script to visualize other files according to your needs.

# Example Usage
You can use the above mentioned scripts (i) to visualize results that were produced using the [evaluation code](../train_and_eval) or (ii) to run the live-demo as shown in the paper. Both modes require that the `inference_server.py` is running.

In the following we assume at least one SMPL model was dragged into the Unity scene to become a component. Please follow the rules explained in the SMPL Unity plugin when performing this. This is important as otherwise vertex IDs will change which can result in unexpected behaviour. When the SMPL model is available as a component, you should see a child called `f_avg` or `m_avg` depending on whether you imported the female or male model. In the following, we refer to this component simply as the SMPL component.

## Live Demo
We recommend to use radio channel 11 at 60 Hz (default values) and the following workflow:

- Start the `inference_server.py` script. A message `Listening on 'IP':'PORT'` should be printed to the console on success.
- Hit the Play button in the Unity project. The Unity script immediately tries to connect to the Awinda master (either the wireless base station or the bluetooth dongle) and prints `Waiting for MTWs to connect to the master.` to the console if successful. If this does not show up, troubleshoot the connection using the MT Manager Software provided by Xsens. Next, the script establishes connection with the inference server and loads the neural network in a TensorFlow Session. Upon success, `Model loaded on server` is printed to the console. If you experience problems, check the console output of the `inference_server.py` process - may be the model files could not be located?
- Now you are ready to connect the 6 sensors to the base station. Remove them from their charging sockets, simply move them around a bit, or press the button located on the sensor to activate them. They now automatically connect to the station. When all 6 sensor's LEDs blink in unison with the base station LED, they're connected and synchronized. You should also see some output on the Unity console that the sensors were detected.
- Once all 6 sensors have connected, you are ready enter to measuring mode, i.e. start streaming data from the sensors. To do so, hit the `Start Measurement` button.
- At this point, the script directly assigns the 6 measured IMU orientations to the respective body segments. Hence, the SMPL model should now move quite weirdly. As long as we are not receiving predictions from the model (which we don't at this point), this is expected and means that data is being streamed successfully.
- You can now mount the sensors onto the subject. If you choose to look at the SMPL model during this process, viewer discretion is strongly advised -- it may not be suitable for viewers under 18. Very importantly, make sure to:
  - **mount the sensors to their correct location on the body**.
  - **mount the head sensor as explained in the paper**. We assume that the head sensor is always mounted in the same local orientation. If this is not the case, the calibration will not be precise and hence the results won't be either.
- Once all sensors are in place, instruct the subject to perform the calibration pose as explained in the paper. Then hit the `Calibrate` button. The `Calibrate` button is only available if the head sensor is levelled, i.e. pitch and roll are within a reasonable range around 0. We implemented this to ensure a certain quality of the calibration. If you don't like this, you can disable it by setting `_calibrationEnabled` in `read_mtw.cs` to true always.
- After the calibration the SMPL model should no longer move weirdly. The script still only displays the orientation for the body segments tracked by sensors, but they should now be "correct". You might want to check that it was successful by telling the subject to do some simple movements.
- Now the system is ready. Hit `Toggle Model Inference` to get the predicted poses from the inference server and enjoy the full output of DIP. Hit it again to disable inference.
- When you are done just stop the Unity project, which will stop all communications gracefully. Make sure to wait at least for some seconds before starting the Unity project again. This way, the sensors cann disconnect from the master. They disconnected successfully when they stop blinking in unison. If you don't wait long enough the Unity script might fail to recognise the sensors on the next startup, which leads to NullReferenceErrors. If this is the case, just restart Unity and plug the dongle or master station out and back in again.

There are a few other buttons appearing on screen when the Unity project is running:
- `Heading Reset`: Performs a heading reset on all connected sensors. We used this when recording the dataset to compensate for the different heading offsets in each sensor. If you want to use this, perform the heading reset before mounting the sensors onto the subject (but after you clicked on `Start Measurement`): Make sure all sensors are aligned, i.e., facing the same way. For example, place them along a side of a box. Then hit the `Heading Reset` button and continue with the mounting procedure.
- `Toggle Acceleration`: Draws the acceleration vector as a Gizmo - for debugging purposes.
- `Draw Bone Orientations`:  Draws the bone orientations as a Gizmo - for debugging purposes.
- `Set I-Pose`: Sets the hard coded calibration pose - for debugging purposes.
- `Dump Recording`: Sends a command to the inference server to dump some recorded data to disk - for debugging purposes, cf. `dump_recordings` in `inference_server.py` for details.

## Visualization
To use the visualization code, add the script `visualize_data.cs` instead of `read_mtw.cs` to the SMPL component. You should see the fields `OURS`, `SIP`, `SOP`, and `GT`. This script was used to produce the figures and videos in the paper, this is why it expects three other SMPL components to be available in the scene. You can create these components or just change the script to use only one.

These four fields are meant to contain names of the files that you want to load (e.g. a `.npz` file created by the evaluation code). The remaining three fields (`SIP Transform`, `SOP Transform`, `GT Transform`) should be mapped to the SMPL component of the three other models (in case you chose to create these as well). E.g., if you created another SMPL component in the scene to show a second motion, just drag it (the `m_avg` or `f_avg`) into one of these fields.

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