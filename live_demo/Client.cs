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
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using UnityEngine;

using System;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;

/// <summary>
/// Serializable class to store a pose in angle-axis format.
/// </summary>
public class Pose {
    // List with 72 entries, i.e. angle-axis representation for 24 joints
    [JsonProperty("pose")]
    public List<float> pose { get; set; }
}

/// <summary>
/// Serializable class to store IMU measurements.
/// </summary>
public class IMUMeasurement {

    public static IMUMeasurement Dummy() {
        IMUMeasurement imu = new IMUMeasurement();

        for (int i = 0; i < 6; i++) {
            imu.orientations[i * 9] = 1.0f;
            imu.orientations[i * 9 + 4] = 1.0f;
            imu.orientations[i * 9 + 4] = 1.0f;
        }

        return imu;
    }

    public IMUMeasurement() : this(9, 6) { }

    public IMUMeasurement(int rotationDOF, int nSensors) {
        orientations = new List<float>(new float[rotationDOF * nSensors]);
        accelerations = new List<float>(new float[3 * nSensors]);
    }

    // List with 54 entries, i.e. 6 rotation matrices
    [JsonProperty("orientations")]
    public List<float> orientations { get; set; }

    // List with 18 entries, i.e. 6 acceleration vectors
    [JsonProperty("accelerations")]
    public List<float> accelerations { get; set; }
}

public delegate void MessageAvailableEventHandler(object sender,
    MessageAvailableEventArgs e);

public class MessageAvailableEventArgs : EventArgs {
    public MessageAvailableEventArgs(string message) : base() {
        this.Message = message;
    }

    public string Message { get; private set; }
}

/// <summary>
/// Simple TCP Client that connects to the Python inference server.
/// </summary>
public class Client {
    private TcpClient socket;
    private NetworkStream stream;
    private String bindIP = "127.0.0.1";
    private int bindPort = 9999;
    private Byte[] sizeBuffer = new Byte[2];
    
    private bool _connectedToServer = false;

    public event MessageAvailableEventHandler MessageAvailable;

    public void ConnectToTcpServer() {
        try {
            socket = new TcpClient();
            socket.NoDelay = true;
            socket.Connect(bindIP, bindPort);
            stream = socket.GetStream();
            _connectedToServer = true;
        }
        catch (Exception e) {
            Debug.Log("On client connect exception " + e);
        }
    }

    public void Close() {
        Debug.Log("closing connection to server");
        stream.Close();
        socket.Close();
    }

    protected void OnMessageAvailable(MessageAvailableEventArgs e) {
        var handler = MessageAvailable;
        if (handler != null)
            handler(this, e);
    }

    public void ListenAsync() {
        if (!_connectedToServer) {
            return;
        }

        // asynchronous read, calls the callback once data is available
        stream.BeginRead(sizeBuffer, 0, sizeBuffer.Length, new AsyncCallback(ReadCallback), null);
        Debug.Log("begin read finished");   
    }

    private void ReadCallback(IAsyncResult ar) {
        // this blocks until the data is ready
        int bytesRead = stream.EndRead(ar);

        Debug.Log("end read finished " + bytesRead);

        if (bytesRead != 2)
            throw new InvalidOperationException("Invalid message header.");


        int messageSize = BitConverter.ToUInt16(sizeBuffer, 0);
        //        int messageSize = BitConverter.ToInt32(sizeBuffer, 0);
        Debug.Log("message size " + messageSize);

        // now read the actual message
        ReadMessage(messageSize);

        // start listening again
        ListenAsync();
    }

    private void ReadMessage(int messageSize) {
        try {

            int remainingLength = messageSize;
            string serverMessage = "";

            Debug.Log("getting message of size " + messageSize);
            
            while (remainingLength > 0) {
                var incomingData = new Byte[remainingLength];
                int bytes_read = stream.Read(incomingData, 0, remainingLength);

                if (bytes_read > 0) {
                    serverMessage += Encoding.ASCII.GetString(incomingData);
                }
                else {
                    Debug.Log("something went wrong when reading data");
                }

                remainingLength -= bytes_read;
            }

            Debug.Log("read message " + serverMessage);

            // do something with the servermessage here
            OnMessageAvailable(new MessageAvailableEventArgs(serverMessage));
        } 

        catch (SocketException socketException) {
            Debug.Log("Socket exception when receiving: " + socketException);
        }

    }

    public string ListenSync() {
        try {

            if (!_connectedToServer) {
                return "";
            }

            Byte[] bytes = new Byte[2];
            string serverMessage = "";

            // Read first 4 bytes for length of the message
            int length = stream.Read(bytes, 0, bytes.Length);
            if (length == 2) {
                int msgLength = BitConverter.ToUInt16(bytes, 0);

                while (msgLength > 0) {
                    var incomingData = new Byte[msgLength];
                    int bytes_read = stream.Read(incomingData, 0, msgLength);
                    // Debug.Log("Msg length read " + bytes_read);

                    if (bytes_read > 0) {
                        serverMessage += Encoding.ASCII.GetString(incomingData);
                    }
                    else {
                        Debug.Log("something went wrong when reading data");
                    }

                    msgLength -= bytes_read;
                }
                

            }
            return serverMessage;
        }
        catch (SocketException socketException) {
            Debug.Log("Socket exception when receiving: " + socketException);
        }
        return "";
    }

    public void SendSync(string msg) {
        if (socket == null) {
            return;
        }
        try {
            if (stream.CanWrite) {
                // Convert string message to byte array
                byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(msg);
                byte[] length = BitConverter.GetBytes((short)clientMessageAsByteArray.Length);

                var all = new byte[length.Length + clientMessageAsByteArray.Length];
                length.CopyTo(all, 0);
                clientMessageAsByteArray.CopyTo(all, length.Length);

                // Write byte array to socketConnection stream.                 
                stream.Write(all, 0, all.Length);
            }
        }
        catch (SocketException socketException) {
            Debug.Log("Socket exception when sending: " + socketException);
        }
    }

}
