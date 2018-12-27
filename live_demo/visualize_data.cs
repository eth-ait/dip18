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
using Newtonsoft.Json;

/// <summary>
/// Helper class to interface with the server.
/// </summary>
public class IMUData {

    // List with 54 entries, i.e. 6 rotation matrices
    [JsonProperty("orientations")]
    public List<float> orientations { get; set; }

    // List with 18 entries, i.e. 6 acceleration vectors
    [JsonProperty("accelerations")]
    public List<float> accelerations { get; set; }
}

/// <summary>
/// Visualizes the contents of a pickle (.pkl) file that contains pose parameters.
/// </summary>
public class RawDataVisualizer {
    
    private Client _client = null;
    public SMPLPoseUpdater _poseUpdater;

    private Mesh _currentMesh;
    private SkinnedMeshRenderer _meshRenderer;
    private IMUData _currentIMUData = null;

    private List<int> _imuVertices = new List<int> { 1962, 5431, 1096, 4583, 412, 3021 };

    public bool DrawAcceleration { get; set; }

    public RawDataVisualizer(string filename, Transform transform) {
        _currentMesh = new Mesh();
        _client = new Client();
        _meshRenderer = transform.GetComponent<SkinnedMeshRenderer>();
        _poseUpdater = new SMPLPoseUpdater(transform, "male");
        _client.ConnectToTcpServer();

        // load the sample on the server
        _client.SendSync("SAMPLELOAD");
        _client.SendSync(filename);
    }

    public void OnApplicationQuit() {
        _client.Close();
    }

    public void NextFrame() {
        _client.SendSync("SAMPLENEXT");
        ReceiveFrame();
    }

    public void PreviousFrame() {
        _client.SendSync("SAMPLEPREV");
        ReceiveFrame();
    }

    private void ReceiveFrame() {
        // first the pose is sent
        string pose = _client.ListenSync();
        Pose p = JsonConvert.DeserializeObject<Pose>(pose);
        _poseUpdater.setNewPose(p.pose.ToArray());

        // then the IMU data is sent
        string imu = _client.ListenSync();
        _currentIMUData = JsonConvert.DeserializeObject<IMUData>(imu);
    }

    public void Draw() {
        if (_currentIMUData == null) return;

        if (_currentIMUData.orientations.Count != 18) {
            throw new UnityException("orientation expected in angle-axis format");
        }

        _meshRenderer.BakeMesh(_currentMesh);

        // Uncomment to draw IMU orientations using cylinders
        //_poseUpdater.DrawIMUBoneOrientationAt(_meshRenderer.transform.position + _currentMesh.vertices[4583]);

        for (int i = 0; i < 6; i++) {
            // set the position
            Vector3 pos = _meshRenderer.transform.position + _currentMesh.vertices[_imuVertices[i]];

            // draw orientation
            Vector3 ori = new Vector3(_currentIMUData.orientations[i * 3],
                                      _currentIMUData.orientations[i * 3 + 1],
                                      _currentIMUData.orientations[i * 3 + 2]);

            // Uncomment to draw IMU orientations using simple Gizmo lines
            // Quaternion quat = Quaternion.AngleAxis(ori.magnitude * Mathf.Rad2Deg, ori.normalized);
            // DrawingHelper.DrawRotationAxes(quat, pos);

            if (DrawAcceleration) {
                Vector3 acc = new Vector3(_currentIMUData.accelerations[i * 3],
                                          _currentIMUData.accelerations[i * 3 + 1],
                                          _currentIMUData.accelerations[i * 3 + 2]);

                DrawingHelper.DrawAcceleration(pos, acc, Color.white);
            }

        }

    }

}

/// <summary>
/// The MonoBehaviour derivative. Assumes that there are four SMPL models available in the scene.
/// </summary>
public class visualize_data : MonoBehaviour {

    public string OURS;

    public string SIP = "";
    public string SOP = "";
    public string GT = "";

    public Transform SIPTransform = null;
    public Transform SOPTransform = null;
    public Transform GTTransform = null;

    private bool _paused = false;
    private List<RawDataVisualizer> _visualizers = new List<RawDataVisualizer>();

    private GameObject plane;
    
	void Start () {
        _visualizers.Add(new RawDataVisualizer(OURS, this.transform));
        if (SIP != "") {
            _visualizers.Add(new RawDataVisualizer(SIP, SIPTransform));
        }
        if (SOP != "") {
            _visualizers.Add(new RawDataVisualizer(SOP, SOPTransform));
        }
        if (GT != "") {
            _visualizers.Add(new RawDataVisualizer(GT, GTTransform));
        }
        plane = GameObject.Find("Plane");
    }

    private void OnApplicationQuit() {
        foreach(RawDataVisualizer v in _visualizers) {
            v.OnApplicationQuit();
        }
    }

    private void OnGUI() {
        if (GUILayout.Button("Toggle Acceleration")) {
            foreach(RawDataVisualizer v in _visualizers) {
                v.DrawAcceleration = !v.DrawAcceleration;
            }
        }

        if(GUILayout.Button("Next Frame")) {
            if(_paused) {
                foreach (RawDataVisualizer v in _visualizers) {
                    v.NextFrame();
                }
            }
        }

        if (GUILayout.Button("Previous Frame")) {
            if(_paused) {
                foreach (RawDataVisualizer v in _visualizers) {
                    v.PreviousFrame();
                }
            }
        }

        if (GUILayout.Button("Toggle Shadows")) {
            plane.GetComponent<Renderer>().receiveShadows = !plane.GetComponent<Renderer>().receiveShadows;
        }
    }

    void Update () {

        if (Input.GetKeyDown(KeyCode.Space)) {
            _paused = !_paused;
        }

        
        foreach (RawDataVisualizer v in _visualizers) {
            if (!_paused) {
                v.NextFrame();
            }
        }
        
        foreach (RawDataVisualizer v in _visualizers) {
            v.Draw();
        }
    }
    
}
