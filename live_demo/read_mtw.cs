/*  DIP: training, evaluating and running of deep inertial poser.
    Copyright (C) 2018 ETH Zurich, Manuel Kaufmann

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
 */
using System.Collections.Generic;
using UnityEngine;
using XDA;
using Xsens;
using Newtonsoft.Json;
using System.Diagnostics;

/// <summary>
/// Main class to drive the connection to Xsens and Python for retrieval and visualization of IMU measurements
/// and predicted SMPL pose parameters.
/// </summary>
public class read_mtw : MonoBehaviour {

    // for communication with master
    private MyXda _xda;
    private XsDevice _masterDevice = null;

    // settings for master
    public int RadioChannel = 11;
    public int Hz = 60;  // max Hz for 1 to 5 sensors: 120, max Hz for 6 sensors: 100
    
    // storing data per connected MTw
    private Dictionary<XsDevice, MyMtwCallback> _measuringMts = new Dictionary<XsDevice, MyMtwCallback>();
    private Dictionary<uint, XsIMUMeasurement> _connectedMtwData = new Dictionary<uint, XsIMUMeasurement>();
    private Dictionary<uint, string> _imuIdToName = new Dictionary<uint, string>();
    private Dictionary<uint, string> _imuIdToBoneName = new Dictionary<uint, string>();
    private Dictionary<uint, uint> _imuIdToVertex = new Dictionary<uint, uint>();

    // sensor IDs
    private static uint _headId = 11809015;
    private static uint _lArmId = 11809000;  // left upper arm sensor
    private static uint _rArmId = 11809024;  // right upper arm sensor
    private static uint _lLegId = 11808713;  // left lower leg sensor 
    private static uint _rLegId = 11808755;  // right lower leg sensor
    private static uint _pelvisId = 11808759;

    // model expects sensors in order l_elbow, r_elbow, l_knee, r_knee, sternum, pelvis
    // sensors used here are: 
    private List<uint> _imuOrder = new List<uint> { _lArmId, _rArmId, _lLegId, _rLegId, _headId, _pelvisId };
    
    // the current SMPL mesh
    private SkinnedMeshRenderer _meshRenderer;
    private Mesh _currentMesh;

    // the client communicating with the inference server
    private Client _client;

    // to set a new pose on the SMPL mesh
    private SMPLPoseUpdater _poseUpdater;

    private bool _modelLoaded = false;
    private bool _acceptNewMTWs = true;
    private bool _MTWsInitialized = false;
    private bool _getModelPrediction = false;
    private bool _drawAcceleration = false;
    private bool _drawBoneOrientations = false;
    private bool _drawIMUOriAsBoneOri = true;
    private bool _calibrationEnabled = false;
    private int _totalConnectedMTWs = 0;

    // for framerate statistics
    private Stopwatch sw;

	void Start () {

        _currentMesh = new Mesh();
        _meshRenderer = GetComponent<SkinnedMeshRenderer>();
        _poseUpdater = new SMPLPoseUpdater(this.transform);
        _client = new Client();

        // must match the name of the object in the scene
        _imuIdToName.Add(_rArmId, "r_arm");
        _imuIdToName.Add(_rLegId, "r_leg");
        _imuIdToName.Add(_lArmId, "l_arm");
        _imuIdToName.Add(_lLegId, "l_leg");
        _imuIdToName.Add(_pelvisId, "pelvis");
        _imuIdToName.Add(_headId, "head");

        // must match name in definition of SMPL model
        _imuIdToBoneName.Add(_rArmId, "R_Elbow");
        _imuIdToBoneName.Add(_rLegId, "R_Knee");
        _imuIdToBoneName.Add(_lArmId, "L_Elbow");
        _imuIdToBoneName.Add(_lLegId, "L_Knee");
        _imuIdToBoneName.Add(_pelvisId, "Pelvis");
        _imuIdToBoneName.Add(_headId, "Head");

        // to determine location where to display IMU sensors
        _imuIdToVertex.Add(_rArmId, 5431);  // right arm
        _imuIdToVertex.Add(_rLegId, 4583);  // right leg
        _imuIdToVertex.Add(_lArmId, 1962);  // left arm
        _imuIdToVertex.Add(_lLegId, 1096);  // left leg
        _imuIdToVertex.Add(_pelvisId, 3021);  // pelvis
        _imuIdToVertex.Add(_headId, 412);

        // intialize control object that handles communication to XSens device
        _xda = new MyXda();

        // scan for devices
        ScanForStations();

        // enable radio for master device found when scanning for stations
        EnableRadio();
        _totalConnectedMTWs = _masterDevice.childCount();

        UnityEngine.Debug.Log("Waiting for MTWs to connect to the master.");

        // connect to inference server
        _client.ConnectToTcpServer();

        // load the model into a TensorFlow session
        LoadModel();
	}

    void OnApplicationQuit() {

        // shut down communication to Xsens
        if (_masterDevice != null) {
            _masterDevice.clearCallbackHandlers();
        }

        _measuringMts.Clear();

        UnityEngine.Debug.Log("Disable radio");
        if (_masterDevice != null) {
            if (_masterDevice.isRadioEnabled()) {
                _masterDevice.disableRadio();
            }
        }

        UnityEngine.Debug.Log("closing connection to MTws");
        _xda.Dispose();
        _xda = null;

        // shut down connection to server
        _client.Close();
    }

    private void ScanForStations() {
        List<MasterInfo> _stations = new List<MasterInfo>();
        _xda.scanPorts();

        if (_xda._DetectedDevices.Count > 0) {
            foreach (XsPortInfo portInfo in _xda._DetectedDevices) {
                if (portInfo.deviceId().isWirelessMaster() || portInfo.deviceId().isAwindaStation()) {
                    UnityEngine.Debug.Log("found wireless connector");
                    _xda.openPort(portInfo);
                    MasterInfo ai = new MasterInfo(portInfo.deviceId());
                    ai.ComPort = portInfo.portName();
                    ai.BaudRate = portInfo.baudrate();
                    _stations.Add(ai);
                    break;
                }
            }

            if (_stations.Count > 0) {
                UnityEngine.Debug.Log("Found station: " + _stations[0].ToString() + " ... creating master device.");
                _masterDevice = _xda.getDevice(_stations[0].DeviceId);

                if(!_masterDevice.gotoConfig()) {
                    throw new UnityException("could not enter configuration mode of created master device");
                }
                UnityEngine.Debug.Log("master device created successfully, ready to enable radio");
            }
            else {
                throw new UnityException("no station could be found, make sure drivers are installed correctly");
            }
        }
    }

    private void EnableRadio() {
        if (!_masterDevice.setUpdateRate(Hz)) {
           throw new UnityException("failed to set update rate " + Hz);
        }

        if (_masterDevice.isRadioEnabled()) {
            _masterDevice.disableRadio();
        }

        if(_masterDevice.enableRadio(RadioChannel)) {
            UnityEngine.Debug.Log("radio enabled successfully");
        }
        else {
            throw new UnityException("could not enable radio");
        }
    }

    private void CheckForMTWConnections() {
        if(_acceptNewMTWs) {
            int nextCount = _masterDevice.childCount();
            if(nextCount != _totalConnectedMTWs) {
                UnityEngine.Debug.Log("Number of connected MTWs: " + nextCount);
                _totalConnectedMTWs = nextCount;

                XsDevicePtrArray deviceIds = _masterDevice.children();
                for (uint i = 0; i < deviceIds.size(); i++) {
                    XsDevice dev = new XsDevice(deviceIds.at(i));
                    UnityEngine.Debug.Log(string.Format("Device {0} ({1})", i, dev.deviceId().toInt()));
                }
            }
        }
    }
	
	void Update () {

        if(_drawBoneOrientations) {
            _poseUpdater.DrawIMUBoneOrientations();
        }

        CheckForMTWConnections();

        if (_acceptNewMTWs) {
            return;
        }

        if (!_MTWsInitialized) {
            _connectedMtwData.Clear();
            if (!_masterDevice.gotoMeasurement()) {
                throw new UnityException("could not enter measurement mode");
            }

            _masterDevice.clearCallbackHandlers();
            XsDevicePtrArray deviceIds = _masterDevice.children();
            for (uint i = 0; i < deviceIds.size(); i++) {

                XsDevice mtw = new XsDevice(deviceIds.at(i));
                MyMtwCallback callback = new MyMtwCallback();
                uint deviceId = mtw.deviceId().toInt();

                if (_imuOrder.Contains(deviceId)) {
                    
                    XsIMUMeasurement mtwData = new XsIMUMeasurement();
                    _connectedMtwData.Add(deviceId, mtwData);

                    callback.DataAvailable += new System.EventHandler<DataAvailableArgs>(DataAvailableCallback);

                    mtw.addCallbackHandler(callback);
                    _measuringMts.Add(mtw, callback);
                }
                
            }

            _MTWsInitialized = true;
            UnityEngine.Debug.Log(string.Format("Initialized {0} MTWs", _measuringMts.Keys.Count));
        }

        if (_MTWsInitialized) {
            // draw IMU measurements in Unity
            // bake mesh so that we can get updated vertex positions
            _meshRenderer.BakeMesh(_currentMesh);
            foreach (KeyValuePair<uint, XsIMUMeasurement> data in _connectedMtwData) {

                if (_drawIMUOriAsBoneOri) {
                    _poseUpdater.setBoneOrientation(_imuIdToBoneName[data.Key], _connectedMtwData[data.Key].quat);
                }

                data.Value.Draw(_meshRenderer.transform.position + _currentMesh.vertices[_imuIdToVertex[data.Key]],
                                _drawAcceleration);

            }

            if (_drawIMUOriAsBoneOri) {
                _poseUpdater.setBoneOrientation("Head", _connectedMtwData[_headId].quat);
            }

            // send IMU measurements to inference server and display the results
            if (_getModelPrediction) {
                GetAndDisplayModelPrediction();
            }
            else {
                // make sure the head sensor is levelled
                // only compute this when model inference not toggled
                float pitch = getPitch(_connectedMtwData[_headId].quat);
                float roll = getRoll(_connectedMtwData[_headId].quat);
                if (Mathf.Abs(pitch) < 5.0f && Mathf.Abs(roll) < 5.0f) {
                    _calibrationEnabled = true;
                    UnityEngine.Debug.Log("Calibration ENABLED");
                }
                else {
                    _calibrationEnabled = false;
                    UnityEngine.Debug.Log("Head sensor not levelled, pitch: " + pitch + " roll: " + roll);
                }
            }
        }
        
    }

    private float getPitch(Quaternion q) {
        Matrix4x4 m = Matrix4x4.Rotate(q);
        float pitch = Mathf.Atan2(-m[2, 0], Mathf.Sqrt(m[2, 1] * m[2, 1] + m[2, 2] * m[2, 2]));
        return pitch * 180.0f / Mathf.PI;
    }

    private float getRoll(Quaternion q) {
        Matrix4x4 m = Matrix4x4.Rotate(q);
        float roll = Mathf.Atan2(m[2, 1], m[2, 2]);
        return roll * 180.0f / Mathf.PI;
    }

    void OnGUI() {
        if (GUILayout.Button("Start Measurement")) {
            _acceptNewMTWs = false;
            UnityEngine.Debug.Log("Starting measurements with " + _totalConnectedMTWs + " MTws ...");
        }

        if (GUILayout.Button("Stop Measurement")) {
            StopMeasurement();
        }

        if (GUILayout.Button("Heading Reset")) {
            HeadingReset();
        }

        if (_calibrationEnabled) {
            if (GUILayout.Button("Calibrate")) {
                Calibrate();
            }
        }

        if (GUILayout.Button("Toggle Model Inference")) {
            _getModelPrediction = !_getModelPrediction;
            _drawIMUOriAsBoneOri = !_drawIMUOriAsBoneOri;
        }

        if (GUILayout.Button("Toggle Acceleration")) {
            _drawAcceleration = !_drawAcceleration;
        }

        if (GUILayout.Button("Draw Bone Orientations")) {
            _drawBoneOrientations = !_drawBoneOrientations;
        }

        if (GUILayout.Button("Set I-Pose")) {
            _poseUpdater.setNewPose(SMPLPoseUpdater.iPose);
        }

        if (GUILayout.Button("Dump Recording")) {
            _client.SendSync("DUMP");
        }

    }

    private void HeadingReset() {
        if (!_MTWsInitialized) {
            return;
        }

        // this should only be called after a few seconds after entering measuring mode
        foreach (KeyValuePair<XsDevice, MyMtwCallback> data in _measuringMts) {
            if (!data.Key.resetOrientation(XsResetMethod.XRM_Heading)) {
                throw new UnityException("could not reset sensor " + data.Key.deviceId());
            }
        }
    }

    private void StopMeasurement() {
        if (!_MTWsInitialized) {
            return;
        }

        _MTWsInitialized = false;
        _acceptNewMTWs = true;

        if (_masterDevice.isRecording()) {
            _masterDevice.stopRecording();
        }
        _masterDevice.gotoConfig();

        _poseUpdater.setNewPose(SMPLPoseUpdater.tPose);

        UnityEngine.Debug.Log("Measurement stopped, " + _masterDevice.childCount() + " MTws connected");
    }

    private void Calibrate() {
        /* The purpose of the calibration is to find the rotation offset between the orientation of the sensor
         * and the orientation of the bone. It assumes that the orientation of the IMU sensor is given in the
         * global (inertial/tracking) frame. The output is the orientation of the sensor in the SMPL body
         * frame.
         * To perform the calibration, the subject must stand still in the i-pose. Then, this method should
         * be invoked.
         */
        if (!_MTWsInitialized) {
            throw new UnityException("Cannot perform calibration, MTWs are not initialized.");
        }

        if(!_connectedMtwData.ContainsKey(_headId)) {
            throw new UnityException("Cannot perform calibration, head sensor not connected.");
        }

        // reset head quaternions in case we need to do calibration multiple times
        _connectedMtwData[_headId].calibrationQuat = Quaternion.identity;
        _connectedMtwData[_headId].boneQuat = Quaternion.identity;

        // quaternion that rotates the sensor axes so that they align with the SMPL frame
        // this assumes that the sensor is placed correctly on the head
        Quaternion R = new Quaternion(0.5f, 0.5f, 0.5f, 0.5f);

        // the quaternion that maps from inertial to SMPL frame
        Quaternion calib = Quaternion.Inverse(_connectedMtwData[_headId].quat * R);

        // set the calibration quat also on the head sensor for debugging
        // note that if we do this, we can only calibrate once when the system is running
        _connectedMtwData[_headId].calibrationQuat = calib;

        // put the model in i-Pose, so that we can read off the correct bone orientations
        _poseUpdater.setNewPose(SMPLPoseUpdater.iPose);

        for (int i = 0; i < _imuOrder.Count; i++) {
            uint imuId = _imuOrder[i];

            if (!_connectedMtwData.ContainsKey(imuId)) {
                continue;
            }
            _connectedMtwData[imuId].calibrationQuat = calib;
            _connectedMtwData[imuId].boneQuat = Quaternion.identity;

            Quaternion qBone = _poseUpdater.GetGlobalBoneOrientation(_imuIdToBoneName[imuId]);
            Quaternion qSensor = _connectedMtwData[imuId].quat;

            _connectedMtwData[imuId].boneQuat = Quaternion.Inverse(qSensor) * qBone;
        }

    }

    private void DataAvailableCallback(object sender, DataAvailableArgs e) {
        _connectedMtwData[e.Device.deviceId().toInt()].OnDataAvailable(e.Packet);
    }

    private void GetAndDisplayModelPrediction() {
        sw = Stopwatch.StartNew();
        IMUMeasurement imu = new IMUMeasurement();

        for (int i = 0; i < _imuOrder.Count; i++) {

            if (_connectedMtwData.ContainsKey(_imuOrder[i])) {
                XsIMUMeasurement data = _connectedMtwData[_imuOrder[i]];

                // get rotation
                Matrix4x4 rot = Matrix4x4.Rotate(data.quat);
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < 3; k++) {
                        int idx = k * 3 + j; // between 0 and 8
                        idx = i * 9 + idx; // between 0 and 53
                        imu.orientations[idx] = rot[k, j];
                    }
                }

                /*if (_imuIdToName[_imuOrder[i]] == "pelvis") {
                    Debug.Log(string.Format("Pelvis rotation sent {0}", rot));
                }*/

                // get acceleration
                imu.accelerations[i * 3] = data.freeAcc.x;
                imu.accelerations[i * 3 + 1] = data.freeAcc.y;
                imu.accelerations[i * 3 + 2] = data.freeAcc.z;
            }
            else {
                UnityEngine.Debug.LogError("Not all IMUs required for model inference initialized");
                // just set identity orientation and zero acceleration
                imu.orientations[i * 9] = 1.0f;
                imu.orientations[i * 9 + 4] = 1.0f;
                imu.orientations[i * 9 + 8] = 1.0f;
            }
        }

        string msg = SendIMUMeasurement(imu);
        Pose p = JsonConvert.DeserializeObject<Pose>(msg);

        if(p.pose.Count == 1 && p.pose[0] < 0.0f) {
            // this is not a valid pose, buffer is not yet full so wait
            UnityEngine.Debug.Log("Wait for Buffer to be filled");
            return;
        }

        // only interested in work performed by the model
        sw.Stop();
        UnityEngine.Debug.Log(string.Format("model inference elapsed time [ms]: {0:0.0000}", sw.Elapsed.TotalMilliseconds));

        _poseUpdater.setNewPose(p.pose.ToArray());
        
    }

    public string SendIMUMeasurement(IMUMeasurement imu) {
        if(!_modelLoaded) {
            throw new UnityException("Model not loaded on the server.");
        }
        
        var imu_s = JsonConvert.SerializeObject(imu);

        // notify the server that the next message contains an IMU measurement
        _client.SendSync("IMU");

        // send the actual IMU data
        _client.SendSync(imu_s);

        // Wait for it to return
        return _client.ListenSync();

    }

    public void LoadModel() {
        // loads the model into a TensorFlow Session on the server
        if (!_modelLoaded) {
            _client.SendSync("STARTUP");
            string response = _client.ListenSync();
            if (response == "READY") {
                UnityEngine.Debug.Log("Model loaded on server");
                _modelLoaded = true;
            }
            else {
                _modelLoaded = false;
                throw new UnityException("Could not load model on the server");
            }
        }
        else {
            UnityEngine.Debug.Log("Model already loaded.");
        }

    }
}