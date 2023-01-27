import numpy as np
import argparse 
import glob
import os
import math
import cv2
import scipy.io as sio
import pickle as pkl
import quaternion as pip_quaternion # The quaternion installed via pip
from psbody.smpl.serialization import load_model

ID_SMPL_0 = [15, 9, 0, 16, 17, 18, 19, 1, 2, 4, 5]
ID_SMPL_3 = [15, 9, 3, 16, 17, 18, 19, 1, 2, 4, 5]
if 1:
	#TC_2_TNT = [5, 6, 9, 10, 0, 2]

	# 4_7, to use for SOP/SIP code
	TC_2_TNT = [9, 10, 5, 6, 0, 2]
else:
	TC_2_TNT = [5, 6, 9, 10, 1, 2, 3, 4, 7, 8]

# Todo
MODEL_PATH = '**************/%s/model.pkl'
model = load_model(MODEL_PATH % 'male')

parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str)
parser.add_argument('--res_path', type=str)
args = parser.parse_args()


def pose2matrix(pose):
        res = []

        model.pose[:] = pose
        for j in ID_SMPL_0:
                res.append(model.A_global[j][:3, :3].r)

        return np.array(res)

def compare_imu_mosh(imu_ori, mosh_pose):
        mosh_ori_0 = pose2matrix(mosh_pose[0])
        imu_ori_0 = imu_ori[0]

        res = []
        total = min(len(imu_ori), len(mosh_pose))
        for idx in range(1, total):
                mosh_ori_i = pose2matrix(mosh_pose[idx])

                imu_ori_i = imu_ori[idx]
                # For each joint
                for j in range(0, len(ID_SMPL_0)):
                        tmp = np.dot(imu_ori_i[j, :, :], imu_ori_0[j, :, :].T)
                        imu_ori_i[j, :, :] = np.dot(tmp, mosh_ori_0[j])

                # Compare
                for j in range(0, len(ID_SMPL_0)):
                        tmp = np.dot(mosh_ori_i[j], imu_ori_i[j].T)
                        tmp = cv2.Rodrigues(tmp)[0]
                        tmp = tmp / np.pi * 180
                        res.append(abs(tmp))

# Todo
MOSH_BASE_PATH='******************/TotalCapture_fr12_Dw00100.000_Sw009500.000_Iw00300.000_Bw00012.500_Pw00002.500'

def quaternion_matrix_bk(quaternion_input, is_w_first):
	quaternion = np.copy(quaternion_input)
	if not is_w_first:
		quaternion = [quaternion[3],quaternion[0],quaternion[1],quaternion[2]]

        a = quaternion[0]
        b = quaternion[1]
        c = quaternion[2]
        d = quaternion[3]

        res = np.array([
                [a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])

        res_2 = np.quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
	res_2 = pip_quaternion.as_rotation_matrix(res_2)
	try:
		assert np.mean(np.square(res_2.flatten() - res.flatten())) < 1e-6
	except:
		print 'Bk, Assertion error!'
	return res

def quaternion_matrix(quaternion_input, is_w_first):
    quaternion = np.copy(quaternion_input)
    if not is_w_first:
		quaternion = [quaternion[3],quaternion[0],quaternion[1],quaternion[2]]

    _EPS = np.finfo(float).eps * 4.0

    q = np.array(quaternion, dtype=np.float32, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    res =  np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

    res_2 = np.quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    res_2 = pip_quaternion.as_rotation_matrix(res_2)
    try:
    	assert np.mean(np.square(res_2.flatten() - res.flatten())) < 1e-6
    except:
	print 'Assertion error!'
    return res

# Read IMU
def read_imu(imu_file_path):
	with open(imu_file_path) as fin:
		all_lines = fin.readlines()
	del all_lines[0]
	del all_lines[::14]

	def f1(line):
		str_list = line.split()[1:]
		num_list = [float(str_num) for str_num in str_list]
		return num_list
	data = map(f1, all_lines)
	data = np.array(data)
	imu_ori = data[:, :4]
	imu_acc = data[:, 4:]
	imu_ori_matrix = map(lambda x: quaternion_matrix(x, True), imu_ori)
	imu_ori_matrix_2 = map(lambda x: quaternion_matrix_bk(x, True), imu_ori)
	return imu_ori_matrix, imu_acc

# Read calibration file
def read_calib(calib_file_path):
	with open(calib_file_path) as fin:
		all_lines = fin.readlines()
	del all_lines[0]
	def f1(line):
		str_list = line.split()[1:]
		num_list = [float(str_num) for str_num in str_list]
		return num_list
	data = map(f1, all_lines)
	data = np.array(data)
	calib_matrix = map(lambda x: quaternion_matrix(x, False), data)
	calib_matrix_2 = map(lambda x: quaternion_matrix_bk(x, False), data)
	return calib_matrix

def process_imu(imu_file_path):
	# Read IMU data
	imu_ori_matrix, imu_acc = read_imu(imu_file_path)

	# Read reference data
	calib_file_path = imu_file_path.replace('_Xsens.sensors', '_calib_imu_ref.txt')
	calib_ref_matrix = read_calib(calib_file_path)
	# Read bone data
	calib_file_path = imu_file_path.replace('_Xsens.sensors', '_calib_imu_bone.txt')
	calib_bone_matrix = read_calib(calib_file_path)
	save_path = calib_file_path.replace(args.folder_path, args.res_path).replace('_calib_imu_bone.txt', '.pkl')
	name_splits = os.path.basename(save_path).split('_')
	mosh_path = os.path.join(MOSH_BASE_PATH, name_splits[0], name_splits[1])
	mosh_gt = pkl.load(open(mosh_path))['poses']
        mosh_ori_0 = pose2matrix(mosh_gt[0])
	

	total = len(imu_ori_matrix)
	print total

	ori_global = []
	acc_global = []
	#rot_y = cv2.Rodrigues(np.array([0, -1 * np.pi/2, 0]))[0]
	rot_y = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]

	#tmp1 = pip_quaternion.from_rotation_matrix(imu_ori_matrix[8])
	#tmp2 = pip_quaternion.from_rotation_matrix(calib_ref_matrix[8])
	for i in range(0, total):
		j = i % 13

		ori_tmp = imu_ori_matrix[i]
		calib_ref_tmp_ori = calib_ref_matrix[j]		
		calib_ref_tmp = np.dot(rot_y, calib_ref_tmp_ori)
		new_ori = np.dot(calib_ref_tmp, ori_tmp)
		ori_global.append(new_ori)			

		acc_tmp = imu_acc[i]
		new_acc = np.dot(new_ori, acc_tmp).flatten() - np.array([0, 9.8707, 0])
		acc_global.append(new_acc)
		
	ori_global = np.array(ori_global)
	ori_global = ori_global.reshape([-1, 13, 3, 3])
	ori_global = ori_global[:, :-2, :, :]
	acc_global = np.array(acc_global)
	acc_global = acc_global.reshape([-1, 13, 3])
	acc_global = acc_global[:, :-2, :]

	total = total / 13
	if total > len(mosh_gt): total = len(mosh_gt)
	print imu_file_path
	print total
	print len(mosh_gt)

	for i in range(1, total):
		mosh_ori_i = pose2matrix(mosh_gt[i])
 
		for j in range(0, 11):
			ori_global[i, j, :, :] = np.dot(np.dot(ori_global[i, j, :, :], ori_global[0, j, :, :].T), mosh_ori_0[j])
			#print np.linalg.norm(cv2.Rodrigues(np.dot(ori_global[i, j, :, :], mosh_ori_i[j].T))[0]) * 180 / np.pi
	
	res = {'ori':ori_global[1:, np.array(TC_2_TNT), :, :], 'acc':acc_global[1:, np.array(TC_2_TNT), :], 'gt': mosh_gt[1:]}
	with open(save_path, 'w') as fout:
		pkl.dump(res, fout)
	
	out_file_path = save_path.replace('.pkl', '.mat')
	sio.savemat(out_file_path, mdict={'data': res})
		
	print 'Finish %s\n' % imu_file_path

if __name__ == '__main__':
	all_imu_files = glob.glob(os.path.join(args.folder_path, '*Xsens.sensors'))
	#all_imu_files = glob.glob(os.path.join(args.folder_path, '*s1_walking2*Xsens.sensors'))
	#map(process_imu, all_imu_files)

	for imu_file in all_imu_files:
		try:
			process_imu(imu_file)
		except Exception as e:
			print e
