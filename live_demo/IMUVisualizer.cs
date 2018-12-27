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
using System;

/// <summary>
/// Stores the IMU measurements as retrieved from the sensors and provides
/// functions to convert them into the format useful for model inference.
/// </summary>
public class XsIMUMeasurement {

    public static readonly float GRAVITY = 9.8707f;

    // Data as it arrives from the sensor
    private XsEuler _eulerOrientation { get; set; }
    private XsMatrix _rotMatrix { get; set; }
    private XsQuaternion _rotQuat { get; set; }
    private XsVector _accXs { get; set; }
    private XsVector _freeAccXs { get; set; }

    // Keeping track of start position and Unity transform object
    private Transform _transform = null;
    private Vector3 _originalPosition = new Vector3(0.0f, 0.0f, 0.0f);

    // Quaternion aligning inertial and SMPL frame
    public Quaternion calibrationQuat;

    // Quaternion correcting for bone orientation
    public Quaternion boneQuat;

    // Quaternion as read from the IMU sensor
    public Quaternion _quat;

    // Acceleration as read from the IMU sensor
    public Vector3 _acc;

    // Free Acceleration as read from the IMU sensor
    private Vector3 _freeAcc;

    // Gravity in inertial frame
    private Vector3 _gravityZ;

    private int _lastPacketId = -1;

    public XsIMUMeasurement() : this(null, new Vector3(0.0f, 0.0f, 0.0f)){ }
    
    public XsIMUMeasurement(Transform transform_) : this(transform_, transform_.position) { }

    public XsIMUMeasurement(Transform transform_, Vector3 position_) {
        if (transform_ != null) {
            _transform = transform_;
            _originalPosition = new Vector3(position_.x, position_.y, position_.z);
            _transform.position = _originalPosition;
        }

        calibrationQuat = Quaternion.identity;
        boneQuat = Quaternion.identity;
        _quat = new Quaternion();
        _acc = new Vector3(0.0f, 0.0f, 0.0f);
        _gravityZ = new Vector3(0.0f, 0.0f, GRAVITY);
    }

    /// <summary>
    /// The calibrated orientation.
    /// </summary>
    public Quaternion quat {
        get { return calibrationQuat * _quat * boneQuat; }
    }

    /// <summary>
    /// The acceleration without any gravity.
    /// </summary>
    public Vector3 freeAcc {
        get {
            Vector3 noG = _quat * _acc - _gravityZ;
            return calibrationQuat * noG;
        }
    }

    /// <summary>
    /// The acceleration without any gravity as reported by the sensor.
    /// </summary>
    public Vector3 freeAccSensor {
        get {
            return _freeAcc;
        }
    }

    /// <summary>
    /// Callback function when data is available.
    /// </summary>
    /// <param name="packet"></param>
    public void OnDataAvailable(XsDataPacket packet) {
        _eulerOrientation = packet.orientationEuler();
        _rotMatrix = packet.orientationMatrix();
        _rotQuat = packet.orientationQuaternion();
        _accXs = packet.calibratedAcceleration();
        _freeAccXs = packet.freeAcceleration();

        // store rotation and acceleration in Unity friendly format
        _quat = new Quaternion((float)_rotQuat.x(), (float)_rotQuat.y(), (float)_rotQuat.z(), (float)_rotQuat.w());
        _acc = new Vector3((float)_accXs.value(0), (float)_accXs.value(1), (float)_accXs.value(2));
        _freeAcc = new Vector3((float)_freeAccXs.value(0), (float)_freeAccXs.value(1), (float)_freeAccXs.value(2));

        int currentPacketId = packet.packetCounter();
        if(_lastPacketId == -1) {
            // this is the first frame
            _lastPacketId = currentPacketId;
        }

        int packetDiff = currentPacketId - _lastPacketId;
        if (packetDiff > 10) {
            Debug.Log("packet loss " + packetDiff);
        }
        _lastPacketId = currentPacketId;
    }

    /// <summary>
    /// Helper function to draw orientation and acceleration.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="drawAcc"></param>
    public void Draw(Vector3 position, bool drawAcc) {
        if (_transform != null && _rotQuat != null) {
            // update position
            _transform.position = new Vector3(0.0f, 0.0f, 0.0f);

            // draw axis
            DrawingHelper.DrawRotationAxes(quat, position);

            if (drawAcc) {
                DrawingHelper.DrawAcceleration(position, freeAcc, Color.white);
                DrawingHelper.DrawAcceleration(position, freeAccSensor, Color.blue);
            }
        }
    }

}