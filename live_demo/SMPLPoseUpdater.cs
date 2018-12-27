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

/// <summary>
/// Helper to find a child of a transform. Thanks to the Unity community for this implementation.
/// https://answers.unity.com/questions/799429/transformfindstring-no-longer-finds-grandchild.html
/// </summary>
public static class TransformRecursiveChildExtension {
    //Breadth-first search
    public static Transform FindChildRecursive(this Transform aParent, string aName) {
        var result = aParent.Find(aName);
        if (result != null)
            return result;
        foreach (Transform child in aParent) {
            result = child.FindChildRecursive(aName);
            if (result != null)
                return result;
        }
        return null;
    }
}

/// <summary>
/// Helper class to draw fancy orientations (using cylinders) or simple orientations
/// and accelerations (using Gizmos).
/// </summary>
public class DrawingHelper {

    private GameObject _cylinderX = null;
    private GameObject _cylinderY = null;
    private GameObject _cylinderZ = null;

    public DrawingHelper() {}

    /// <summary>
    /// Creates the cylinders if they don't exist already
    /// </summary>
    public void prepare() {
        if (_cylinderX == null) {
            _cylinderX = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            _cylinderX.transform.position = new Vector3(0, 0, 0);
            _cylinderX.transform.localScale = new Vector3(.1f, .1f, .1f);
            _cylinderX.GetComponent<Renderer>().material.color = Color.red;
        }

        if (_cylinderY == null) {
            _cylinderY = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            _cylinderY.transform.position = new Vector3(0, 0, 0);
            _cylinderY.transform.localScale = new Vector3(.1f, .1f, .1f);
            _cylinderY.GetComponent<Renderer>().material.color = Color.green;
        }

        if (_cylinderZ == null) {
            _cylinderZ = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            _cylinderZ.transform.position = new Vector3(0, 0, 0);
            _cylinderZ.transform.localScale = new Vector3(.1f, .1f, .1f);
            _cylinderZ.GetComponent<Renderer>().material.color = Color.blue;
        }
    }

    /// <summary>
    /// Draw orientation using Gizmos.
    /// </summary>
    /// <param name="quat"></param>
    /// <param name="pos"></param>
    public static void DrawRotationAxes(Quaternion quat, Vector3 pos) {
        Matrix4x4 rot = Matrix4x4.Rotate(quat);

        // negate X because Unity is left-handed
        float lh_correction = -1.0f;
        Vector3 x = new Vector3(lh_correction * rot.GetColumn(0).x, rot.GetColumn(0).y, rot.GetColumn(0).z) * 0.2f;
        Vector3 y = new Vector3(lh_correction * rot.GetColumn(1).x, rot.GetColumn(1).y, rot.GetColumn(1).z) * 0.2f;
        Vector3 z = new Vector3(lh_correction * rot.GetColumn(2).x, rot.GetColumn(2).y, rot.GetColumn(2).z) * 0.2f;

        Debug.DrawLine(pos, pos + x, Color.red);
        Debug.DrawLine(pos, pos + y, Color.green);
        Debug.DrawLine(pos, pos + z, Color.blue);
    }

    /// <summary>
    /// Draw fancy orientations using cylinders.
    /// </summary>
    /// <param name="quat"></param>
    /// <param name="pos"></param>
    public void DrawRotationCylinder(Quaternion quat, Vector3 pos) {
        Matrix4x4 rot = Matrix4x4.Rotate(quat);

        // negate X because Unity is left-handed
        float lh_correction = -1.0f;
        Vector3 x = new Vector3(lh_correction * rot.GetColumn(0).x, rot.GetColumn(0).y, rot.GetColumn(0).z);
        Vector3 y = new Vector3(lh_correction * rot.GetColumn(1).x, rot.GetColumn(1).y, rot.GetColumn(1).z);
        Vector3 z = new Vector3(lh_correction * rot.GetColumn(2).x, rot.GetColumn(2).y, rot.GetColumn(2).z);

        updateCylinder(pos, x, _cylinderX);
        updateCylinder(pos, y, _cylinderY);
        updateCylinder(pos, z, _cylinderZ);
    }

    /// <summary>
    /// Set position and offset for a cylinder.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="offset"></param>
    /// <param name="cylinder"></param>
    private void updateCylinder(Vector3 pos, Vector3 offset, GameObject cylinder) {
        var length_scale = 5.0f;
        var scale = new Vector3(0.025f, offset.magnitude / length_scale, 0.025f);
        var position = pos + (offset / length_scale);
        position.z = position.z + 0.1f;

        cylinder.transform.position = position;
        cylinder.transform.up = offset;
        cylinder.transform.localScale = scale;
    }

    /// <summary>
    /// Draw accelerations using gizmos.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="acc"></param>
    /// <param name="c"></param>
    public static void DrawAcceleration(Vector3 pos, Vector3 acc, Color c) {
        Debug.DrawLine(pos, pos + acc, c);
    }

    /// <summary>
    /// Convert SMPL coordinate frame to Unity. This is necessary because they use
    /// different conventions. SMPL assumes y is up, x is towards the left arm of the body
    /// and z is facing forward.
    /// </summary>
    /// <param name="quat"></param>
    /// <returns></returns>
    public static Quaternion SMPLToUnity(Quaternion quat) {
        return new Quaternion(-quat.x, quat.y, quat.z, -quat.w);
    }

    /// <summary>
    /// Convert Unity coordinate frame to SMPL coordinate frame.
    /// </summary>
    /// <param name="quat"></param>
    /// <returns></returns>
    public static Quaternion UnityToSMPL(Quaternion quat) {
        return new Quaternion(-quat.x, quat.y, quat.z, -quat.w);
    }
}

/// <summary>
/// Update the pose of a loaded SMPL mesh.
/// </summary>
public class SMPLPoseUpdater {

    private Dictionary<string, int> _boneNameToJointIndex;
    private Dictionary<string, Transform> _boneNameToTransform;
    private List<string> _isIMUBone;
    private DrawingHelper _drawingHelper;

    private SkinnedMeshRenderer targetMeshRenderer;

    private string _gender;
    private string _boneNamePrefix;
    private string _scope;
    public static readonly int numJoints = 24;

    // The hard-coded calibration pose.
    public static readonly float[] iPose = {0.00000000e+00f,  0.00000000e+00f,  0.00000000e+00f,  2.40190000e-02f,
                                            -5.65980000e-02f, -6.00000000e-02f,  4.01920000e-03f,  1.07109000e-01f,
                                            4.00000000e-02f, -6.85148000e-02f,  2.97456624e-02f,  0.00000000e+00f,
                                            -1.50640000e-02f,  1.22855000e-01f, -2.80000000e-03f, -2.51200000e-04f,
                                            -7.49550000e-02f,  2.80000000e-03f, -1.97083023e-02f, -5.90451714e-02f,
                                            0.00000000e+00f, -3.69410000e-02f, -1.39870000e-02f,  1.09700000e-03f,
                                            3.08240000e-02f,  1.10824000e-01f,  5.58300000e-02f,  3.68217919e-02f,
                                            -9.79798425e-03f,  0.00000000e+00f,  7.38820000e-02f,  8.71628260e-02f,
                                            1.15933226e-01f, -1.36454340e-02f,  7.27977859e-02f, -2.04008074e-01f,
                                            2.75226449e-02f,  3.74526146e-02f, -3.26716395e-02f,  7.95110800e-02f,
                                            1.55932400e-02f, -3.61916400e-01f,  7.95110800e-02f, -1.55932400e-02f,
                                            3.61916400e-01f,  4.14048214e-02f, -5.75496269e-03f,  6.12744933e-02f,
                                            -1.08706800e-01f, -1.39227600e-02f, -1.10823788e+00f,  7.96932000e-02f,
                                            2.02324166e-01f,  1.06021472e+00f,  1.14999360e-01f, -1.25600000e-01f,
                                            -1.25600000e-01f,  5.21993600e-02f,  1.25600000e-01f,  1.25600000e-01f,
                                            1.34247560e-01f, -9.28749200e-02f, -8.79514000e-02f,  1.31183097e-02f,
                                            4.85928009e-02f,  6.31077200e-02f, -2.00966541e-01f, -3.42684870e-02f,
                                            -1.76926440e-01f, -1.28807464e-01f,  1.02772092e-01f,  2.61631080e-01f };

    public static readonly float[] tPose = new float[SMPLPoseUpdater.numJoints * 3];

    public float[] currentPose = new float[SMPLPoseUpdater.numJoints * 3];

    public SMPLPoseUpdater(Transform transform) : this(transform, "male") { }
    
    public SMPLPoseUpdater(Transform transform, string gender_) {
        targetMeshRenderer = transform.GetComponent<SkinnedMeshRenderer>();
        _gender = gender_;
        

        if (_gender == "male") {
            _boneNamePrefix = "m_avg_";
        }
        else {
            _boneNamePrefix = "f_avg_";
        }

        _boneNameToJointIndex = new Dictionary<string, int>();

        _boneNameToJointIndex.Add("Pelvis", 0);
        _boneNameToJointIndex.Add("L_Hip", 1);
        _boneNameToJointIndex.Add("R_Hip", 2);
        _boneNameToJointIndex.Add("Spine1", 3);
        _boneNameToJointIndex.Add("L_Knee", 4);
        _boneNameToJointIndex.Add("R_Knee", 5);
        _boneNameToJointIndex.Add("Spine2", 6);
        _boneNameToJointIndex.Add("L_Ankle", 7);
        _boneNameToJointIndex.Add("R_Ankle", 8);
        _boneNameToJointIndex.Add("Spine3", 9);
        _boneNameToJointIndex.Add("L_Foot", 10);
        _boneNameToJointIndex.Add("R_Foot", 11);
        _boneNameToJointIndex.Add("Neck", 12);
        _boneNameToJointIndex.Add("L_Collar", 13);
        _boneNameToJointIndex.Add("R_Collar", 14);
        _boneNameToJointIndex.Add("Head", 15);
        _boneNameToJointIndex.Add("L_Shoulder", 16);
        _boneNameToJointIndex.Add("R_Shoulder", 17);
        _boneNameToJointIndex.Add("L_Elbow", 18);
        _boneNameToJointIndex.Add("R_Elbow", 19);
        _boneNameToJointIndex.Add("L_Wrist", 20);
        _boneNameToJointIndex.Add("R_Wrist", 21);
        _boneNameToJointIndex.Add("L_Hand", 22);
        _boneNameToJointIndex.Add("R_Hand", 23);

        _boneNameToTransform = new Dictionary<string, Transform>();

        foreach (var item in _boneNameToJointIndex) {
            var _boneName = _boneNamePrefix + item.Key;
            Transform t = transform.parent.transform.FindChildRecursive(_boneName);
            _boneNameToTransform.Add(item.Key, t);
        }

        _isIMUBone = new List<string> { "Pelvis", "L_Knee", "R_Knee", "R_Elbow", "L_Elbow", "Spine3" };

        // create cylinders to draw rotation axes if required
        _drawingHelper = new DrawingHelper();
    }

    /// <summary>
    /// Get global bone orientation of a given bone.
    /// </summary>
    /// <param name="boneName"></param>
    /// <returns></returns>
    public Quaternion GetGlobalBoneOrientation(string boneName) {
        return DrawingHelper.UnityToSMPL(_boneNameToTransform[boneName].rotation);
    }

    /// <summary>
    /// Draw all bone orientations.
    /// </summary>
    public void DrawBoneOrientations() {
        foreach(KeyValuePair<string, Transform> bone in _boneNameToTransform) {
            Quaternion quat = bone.Value.transform.rotation;
            DrawingHelper.DrawRotationAxes(quat, bone.Value.transform.position);
        }
    }

    /// <summary>
    /// Draw only the bone orientations where we attached a IMU.
    /// </summary>
    public void DrawIMUBoneOrientations() {
        // a hack to get the right leg IMU to be displayed
        /*Transform _rightLeg = _boneNameToTransform["R_Knee"];
        Quaternion quat = _rightLeg.transform.rotation;
        quat = DrawingHelper.UnityToSMPL(quat);
        DrawingHelper.DrawRotationAxes(quat, _rightLeg.transform.position);
        */
        
        foreach (KeyValuePair<string, Transform> bone in _boneNameToTransform) {
            if (_isIMUBone.Contains(bone.Key)) {
                Quaternion quat = bone.Value.transform.rotation;
                quat = DrawingHelper.UnityToSMPL(quat);
                DrawingHelper.DrawRotationAxes(quat, bone.Value.transform.position);
            }
        }
        
    }

    /// <summary>
    /// Same as DrawIMUBoneOrientations but the rotation axes will be drawn at the given position.
    /// </summary>
    /// <param name="position"></param>
    public void DrawIMUBoneOrientationAt(Vector3 position) {
        _drawingHelper.prepare();
        Transform _rightLeg = _boneNameToTransform["R_Knee"];
        Quaternion quat = _rightLeg.transform.rotation;
        quat = DrawingHelper.UnityToSMPL(quat);
        _drawingHelper.DrawRotationCylinder(quat, position);
    }

    /// <summary>
    /// Set the orientation of the specified bone to the given orientation.
    /// </summary>
    /// <param name="boneName"></param>
    /// <param name="quat"></param>
    public void setBoneOrientation(string boneName, Quaternion quat) {
        Transform trans;
        if (_boneNameToTransform.TryGetValue(boneName, out trans)) {
            trans.rotation = DrawingHelper.SMPLToUnity(quat);
        }
        else {
            Debug.LogError("ERROR: no game object for given bone name: " + boneName);
        }
    }

    /// <summary>
    /// Set a new pose parametrized as a 72 dimensional vector.
    /// </summary>
    /// <param name="pose"></param>
    public void setNewPose(float[] pose) {
        currentPose = pose;
        Quaternion quat;
        int pelvisIndex = -1;
        var _bones = targetMeshRenderer.bones;

        for (int i = 0; i < _bones.Length; i++) {
            int index;
            string boneName = _bones[i].name;
            Transform go;
            // Remove f_avg/m_avg prefix
            boneName = boneName.Replace(_boneNamePrefix, "");

            if (boneName == "root") {
                continue;
            }

            if (boneName == "Pelvis")
                pelvisIndex = i;

            if (_boneNameToJointIndex.TryGetValue(boneName, out index)) {
                int idx = index * 3;
                Vector3 rot = new Vector3(pose[idx], pose[idx + 1], pose[idx + 2]);
                float angle = rot.magnitude * Mathf.Rad2Deg;
                quat = Quaternion.AngleAxis(angle, rot.normalized);

                //Debug.Log(string.Format("{1} rotation {0}", rot, boneName));

                if (_boneNameToTransform.TryGetValue(boneName, out go)) {
                    go.localRotation = DrawingHelper.SMPLToUnity(quat);
                }
                else {
                    Debug.LogError("ERROR: no game object for given bone name: " + boneName);
                }
            }
            else {
                Debug.LogError("ERROR: No joint index for given bone name: " + boneName);
            }
        }
    }

}
