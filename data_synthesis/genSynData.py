'''
Generate synthesis IMU data from H3.6M MoShed results
'''
import glob
import os
import cv2
import sys
# TODO
sys.path.append('./SMPL_python_v.1.0.0/smpl')
sys.path.append('./SMPL_python_v.1.0.0/smpl/smpl_webuser')

import numpy as np
import chumpy as ch
import pickle as pkl

from smpl_webuser.serialization import load_model
from smpl_webuser.lbs import global_rigid_transformation

MODEL_PATH = './SMPL_python_v.1.0.0/smpl/models/basicModel_%s_lbs_10_207_0_v1.0.0.pkl'
model_male = load_model(MODEL_PATH % 'm')
model_female = load_model(MODEL_PATH % 'f')

Jdirs_male = np.dstack([model_male.J_regressor.dot(model_male.shapedirs[:,:,i]) for i in range(10)])
Jdirs_female = np.dstack([model_female.J_regressor.dot(model_male.shapedirs[:,:,i]) for i in range(10)])

# TODO
# Please modify here to specify which vertices to use
VERTEX_IDS = [1962, 5431, 1096, 4583, 412, 3021]

# TODO
TARGET_FPS = 60

# TODO
# Please modify here to specify which SMPL joints to use
SMPL_IDS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

# Get orieentation and acceleraiotn from list of 4x4 matrices, nad vertices
def get_ori_accel(A_global_list, vertex, frame_rate):
    orientation = []
    acceleration = []

    for a_global in A_global_list:
        ori_left_arm = a_global[18][:3, :3].r
        ori_right_arm = a_global[19][:3, :3].r
        ori_left_leg = a_global[4][:3, :3].r
        ori_right_leg = a_global[5][:3, :3].r
        ori_head = a_global[15][:3, :3].r
        ori_root = a_global[0][:3, :3].r

        ori_tmp = []
        ori_tmp.append(ori_left_arm)
        ori_tmp.append(ori_right_arm)
        ori_tmp.append(ori_left_leg)
        ori_tmp.append(ori_right_leg)
        ori_tmp.append(ori_head)
        ori_tmp.append(ori_root)
        
        orientation.append(np.array(ori_tmp))

    time_interval = 1.0 / frame_rate
    total_number = len(A_global_list)
    for idx in range(1, total_number-1):
        vertex_0 = vertex[idx-1].astype(float) # 6*3
        vertex_1 = vertex[idx].astype(float)
        vertex_2 = vertex[idx+1].astype(float)
        accel_tmp = (vertex_2 + vertex_0 - 2*vertex_1) / (time_interval*time_interval)

        acceleration.append(accel_tmp)

    return orientation[1:-1], acceleration


def compute_imu_data(gender, betas, poses, frame_rate):
    if gender == 'male':
        Jdirs = Jdirs_male
        model = model_male
    else:
        Jdirs = Jdirs_female
        model = model_female

    betas[:] = 0
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(model.v_template.r)

    A_global_list = []
    print( 'Length of poses: %d' % (len(poses)/1) )
    for idx, p in enumerate(poses):
        (_, A_global) = global_rigid_transformation(p, J_onbetas, model.kintree_table, xp=ch)
        A_global_list.append(A_global)
    
    vertex = []
    for idx, p in enumerate(poses):
        model.pose[:] = p
        model.betas[:] = 0
        model.betas[:10] = betas
        tmp =  model.r[VERTEX_IDS]
        vertex.append(tmp) # 6*3
            

    orientation, acceleration = get_ori_accel(A_global_list, vertex, frame_rate)

    return orientation, acceleration


def findNearest(t, t_list):
    list_tmp = np.array(t_list) - t
    list_tmp = np.abs(list_tmp)
    index = np.argsort(list_tmp)[:2]
    return index
	

# Turn MoCap data into 60FPS
def interpolation_integer(poses_ori, fps):
    poses = []
    n_tmp = int(fps / TARGET_FPS)
    poses_ori = poses_ori[::n_tmp]
    
    for t in poses_ori:
        poses.append(t)

    return poses

def interpolation(poses_ori, fps):
    poses = []
    total_time = len(poses_ori) / fps
    times_ori = np.arange(0, total_time, 1.0 / fps)
    times = np.arange(0, total_time, 1.0 / TARGET_FPS)
    
    for t in times:
        index = findNearest(t, times_ori)
        a = poses_ori[index[0]]
        t_a = times_ori[index[0]]
        b = poses_ori[index[1]]
        t_b = times_ori[index[1]]

        if t_a == t: 
            tmp_pose = a
        elif t_b == t:
            tmp_pose = b
        else:
            tmp_pose = a + (b-a)*((t_b-t)/(t_b-t_a)) 
        poses.append(tmp_pose)

    return poses


# Extract pose parameter from pkl_path, save to res_path
def generate_data(pkl_path, res_path):
    if os.path.exists(res_path):
        return

    with open(pkl_path) as fin:
        data_in = pkl.load(fin)
    
    data_out = {}
    data_out['gender'] = data_in['gender']
    data_out['betas'] = np.array(data_in['betas'][:10])

    # In case the original frame rates (eg 40FPS) are different from target rates (60FPS) 
    fps_ori = data_in['frame_rate'] 
    if (fps_ori % TARGET_FPS) == 0:
        data_out['poses'] = interpolation_integer(data_in['poses'], fps_ori)
    else:
        data_out['poses'] = interpolation(data_in['poses'], fps_ori)

    data_out['ori'], data_out['acc'] = compute_imu_data(data_out['gender'], data_out['betas'], data_out['poses'], TARGET_FPS)
    
    data_out['poses'] = data_out['poses'][1:-1]

    for fdx in range(0, len(data_out['poses'])):
        pose_tmp = []#np.zeros(0)
        for jdx in SMPL_IDS:
            tmp = data_out['poses'][fdx][jdx*3:(jdx+1)*3]
            tmp = cv2.Rodrigues(tmp)[0].flatten().tolist()
            pose_tmp = pose_tmp + tmp

        data_out['poses'][fdx] = []
        data_out['poses'][fdx] = pose_tmp


    with open(res_path, 'w') as fout:
            pkl.dump(data_out, fout)
    print( pkl_path )
    print( res_path )
    print( len(data_out['acc']) )
    print( '' )
	


# Generate synthesic data for H3.6M
def main(pkl_path, res_data_path):
    generate_data(pkl_path, res_data_path)

if __name__ == '__main__':
    pkl_path = sys.argv[1]	
    res_data_path = sys.argv[2]

    main(pkl_path, res_data_path)
