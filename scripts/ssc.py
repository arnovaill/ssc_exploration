#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Header
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import PointCloud,PointCloud2, ChannelFloat32, Image
from geometry_msgs.msg import Point32, PoseStamped
import pcl 
import tf
import message_filters
import sys
sys.path.append("../SSC")
from models import make_model
import numpy as np
import os
import datetime
from tqdm import tqdm
import ros_numpy # package for conversion ros_msg to numpy
import time
import torch 
from torch.autograd import Variable
from torchvision import transforms
import h5py
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

node_name = 'ssc'
matlab = False

if matlab: 
    import matlab.engine

save_data = False
save_path = '/home/arno2/py3_ws/src/ssc/SSC/data'
weight_file = '/home/arno2/py3_ws/src/ssc/SSC/weights/cpBest_SSC_DDRNet3.pth.tar'

depth_image_topic = '/unreal/unreal_sensor_model/ue_depth_image_out'
rgb_image_topic = '/unreal/unreal_sensor_model/ue_color_image_out'
pointcloud_topic = '/unreal/unreal_sensor_model/ue_sensor_out'
camera_pose_topic = '/unreal/unreal_sensor_model/camera_pose'

# Declare global variables to be updated in callback
counter = np.int64(0) 

# Starting position variables
START_HEIGHT = 1.0

NO_WALL_NORMAL = -1000


colorMap = np.array([[22, 191, 206],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling
                     [43, 160, 4],      # 2 floor
                     [158, 216, 229],   # 3 wall
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     [153, 153, 153],     # 12 label==255, ignore
                     ]).astype(np.int32)

param = {'voxel_size': (240, 144, 240),
            'voxel_unit': 0.02,            # 0.02m, length of each grid == 20mm
            'cam_k': [[518.8579, 0, 320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
                    [0, 518.8579, 240],  # cx = K(1,3); cy = K(2,3);
                    [0, 0, 1]],          # fx = K(1,1); fy = K(2,2);
            'downscale': 4,
            }

pred_ids_camera_frame = []
for i in range (int(param['voxel_size'][0]/param['downscale'])):
    for j in range (int(param['voxel_size'][2]/param['downscale'])):
        for k in range (int(param['voxel_size'][1]/param['downscale'])):
            pred_ids_camera_frame.append([i,j,k])

pred_ids_camera_frame = np.array(pred_ids_camera_frame)
pred_poses_camera_frame = pred_ids_camera_frame*param['voxel_unit']*param['downscale']

transforms_rgb = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

def depth2position(depth, cam_pose, vox_origin):
    cam_k = param['cam_k']
    voxel_size = param['voxel_size']  # (240, 144, 240)
    unit = param['voxel_unit']  # 0.02
    # ---- Get point in camera coordinate
    H, W = depth.shape
    gx, gy = np.meshgrid(range(W), range(H))
    pt_cam = np.zeros((H, W, 3), dtype=np.float32)
    pt_cam[:, :, 0] = (gx - cam_k[0][2]) * depth / cam_k[0][0]  # x
    pt_cam[:, :, 1] = (gy - cam_k[1][2]) * depth / cam_k[1][1]  # y
    pt_cam[:, :, 2] = depth  # z, in meter
    # ---- Get point in world coordinate
    p = cam_pose
    pt_world = np.zeros((H, W, 3), dtype=np.float32)
    pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
    pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
    pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
    pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
    pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
    pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
    # ---- Aline the coordinates with labeled data (RLE .bin file)
    pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
    # pt_world2 = pt_world
    pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
    pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
    pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

    # ---- World coordinate to grid/voxel coordinate
    point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
    point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)

    position = np.zeros((H, W), dtype=np.int32)
    EMPTY = np.empty(voxel_size)

    mask = (point_grid[:,:,0]< voxel_size[0]) * (point_grid[:,:,0] >= 0) * \
            (point_grid[:,:,1]< voxel_size[1]) * (point_grid[:,:,1] >= 0) * \
            (point_grid[:,:,2]< voxel_size[2]) * (point_grid[:,:,2] >= 0)
    coords_image = np.argwhere(mask == True)

    occupied_voxels = [point_grid[coord_image[0], coord_image[1], :] for coord_image in coords_image]
    
    if len(occupied_voxels) == 0:
        position = np.zeros(depth.shape)
    else:
        position[mask] = np.matmul(EMPTY.strides,np.transpose(occupied_voxels))/8

    del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid, mask, coords_image,EMPTY,occupied_voxels  # Release Memory
    return position

def filter_pointcloud(pc):
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    cloud = pcl.PointCloud(np.array(points, dtype=np.float32))
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(0.025, 0.025, 0.025)
    cloud_filtered = sor.filter()
    cloud_filtered_np = cloud_filtered.to_array()
    # filter end round part of PC 
    cloud_filtered_np = cloud_filtered_np[np.sqrt(cloud_filtered_np[:,0]**2 + cloud_filtered_np[:,1]**2 + cloud_filtered_np[:,2]**2) < 5.9]

    return  cloud_filtered_np


def get_wall_normal_angle(cloud_filtered_np):  
    ## whole function get around 0.1s to execute ##
    cloud_filtered = pcl.PointCloud(cloud_filtered_np)
    
    # compute normals
    ne = cloud_filtered.make_NormalEstimation()
    ne.set_RadiusSearch(0.1)
    normals = ne.compute()

    # Keep only normals where curvature is zero
    normals_np = normals.to_array()
    normals_no_curvature = normals_np[normals_np[:,-1] < 0.001]

    # keep only horizontal normals
    normals_horizontal = normals_no_curvature[np.abs(normals_no_curvature[:,1])<0.01][:,:3]

    # make normal point towards surface
    normals_horizontal[:,2] = np.abs(normals_horizontal[:,2])  

    if normals_horizontal.shape[0] > 0:
        horizontal_angles = np.tan(normals_horizontal[:,0]/normals_horizontal[:,2])
        # round angles and evaluate most presents
        horizontal_angles = np.round(horizontal_angles,3)
        values, counts = np.unique(horizontal_angles,return_counts=True,axis=0)
        id_most_present = np.flip(np.argsort(counts))[0]

        wall_normal_angle = values[id_most_present]

        if wall_normal_angle < -np.pi/2 or wall_normal_angle > np.pi/2:
            wall_normal_angle = NO_WALL_NORMAL 

    else:
        wall_normal_angle = NO_WALL_NORMAL    
    
    return wall_normal_angle


def callback(depth_data, rgb_data, pointcloud_data, camera_pose_data):
    global counter
    print('SSC callback rostime:',rospy.get_time())
    depth = ros_numpy.numpify(depth_data)
    print('MAXMAXMAX:', np.max(depth))
    rgb = ros_numpy.numpify(rgb_data)
    pc = ros_numpy.numpify(pointcloud_data)
    camera_world_frame = ros_numpy.numpify(camera_pose_data.pose)

    # Filter pointcloud 
    filtered_pc_np =  filter_pointcloud(pc)

    # Compute wall normal angle 
    wall_normal_angle = get_wall_normal_angle(filtered_pc_np)

    #### Aligned camera pose ####
    if wall_normal_angle != NO_WALL_NORMAL:

        camera_orientation = camera_world_frame.copy()
        camera_orientation[0,3] = 0
        camera_orientation[1,3] = 0
        camera_orientation[2,3] = 0
        camera_aligned = tf.transformations.compose_matrix(angles =(0,wall_normal_angle,0), translate=(0, 0, 0))


        # Find extreme x pointcloud values in aligned camera frame 
        cloud_filtered_aligned = np.array([np.matmul(camera_aligned[:3,:3],point) for point in filtered_pc_np])
        max_pc_aligned = np.max(cloud_filtered_aligned,axis=0)
        min_pc_aligned = np.min(cloud_filtered_aligned,axis=0)

        # Convert extreme values back to camera frame 
        max_pc = camera_aligned[:3,:3]@max_pc_aligned
        min_pc = camera_aligned[:3,:3]@min_pc_aligned

        pred_x = min_pc[0]
        pred_y = START_HEIGHT + tf.transformations.translation_from_matrix(camera_world_frame)[2] 
        pred_z = np.min(filtered_pc_np,axis=0)[-1]*np.cos(wall_normal_angle) - 0.1
        pred_translation = tf.transformations.compose_matrix(angles =(0,0,0), translate=(pred_x,pred_y,pred_z ))
        orientation_correct = tf.transformations.compose_matrix(angles =(np.pi/2,0,0), translate=(0, 0, 0))
        pred_origin = np.linalg.inv(camera_aligned)@pred_translation@orientation_correct # in camera frame

        # tf_broadcaster.sendTransform(tf.transformations.translation_from_matrix(np.linalg.inv(camera_aligned)),
        #         tf.transformations.quaternion_from_matrix(np.linalg.inv(camera_aligned)),
        #         depth_data.header.stamp,
        #         "camera_aligned",
        #         "camera")

        vox_origin = np.array([pred_x,pred_z,-0.05])
        cam_pose = np.linalg.inv(pred_origin)
        cam_pose[0,3] = 0
        cam_pose[1,3] = 0
        cam_pose[2,3] = pred_y -0.05

        # Get 2D-3D mapping index
        position  = depth2position(depth, cam_pose, vox_origin)

        # Normalize rgb image
        rgb_tensor = transforms_rgb(rgb)

        var_depth = Variable(torch.unsqueeze(torch.unsqueeze(torch.tensor(depth),0),0).float()).cuda()
        var_rgb = Variable(torch.unsqueeze(rgb_tensor,0).float()).cuda()
        position = torch.tensor(position).long().cuda()

        with torch.no_grad():
            pred = net(x_depth = var_depth, x_rgb=var_rgb, p = position)
        y_pred = pred.cpu().data.numpy()  # CUDA to CPU, Variable to numpy

        predict_completion = np.squeeze(np.argmax(y_pred, axis=1))
        predict_completion = np.swapaxes(predict_completion,1,2)
        predict_completion = np.swapaxes(predict_completion,0,1)

        pred_origin_world_frame = camera_world_frame@pred_origin 

        # ############ ALl voxels ########### 
        # Translate predicted voxels to world coordinate  
        pred_poses_world_frame = pred_origin_world_frame@np.vstack((pred_poses_camera_frame.T,np.ones(np.shape(pred_poses_camera_frame)[0])))
        pred_poses_world_frame = np.array(pred_poses_world_frame.T)[:,:3]
        occupancies = np.array(predict_completion > 0).flatten()

         # # Declaring pointcloud
        pred_pointcloud = PointCloud()

        # Filling pointcloud header
        header = Header()
        header.stamp = depth_data.header.stamp
        header.frame_id = '/world'
        pred_pointcloud.header = header

        channel_r = ChannelFloat32()
        channel_r.name = "r"
        channel_g = ChannelFloat32()
        channel_g.name = "g"
        channel_b = ChannelFloat32()
        channel_b.name = "b"

        # Filling points (SPEEDUP?) 1.4s
        s = time.clock()
        ptn = Point32(0,0,0)
        for pose, occ in zip(pred_poses_world_frame,occupancies):
            #print(pose)
            occ = int(occ)
            ptn = Point32(pose[0], pose[1], pose[2])
            pred_pointcloud.points.append(ptn)
            channel_r.values.append(colorMap[occ][0])
            channel_g.values.append(colorMap[occ][1])
            channel_b.values.append(colorMap[occ][2])
        pred_pointcloud.channels.append(channel_r)
        pred_pointcloud.channels.append(channel_g)
        pred_pointcloud.channels.append(channel_b)
        print('time to FILL PC: ', time.clock()- s)

        # Publish TF
        tf_broadcaster.sendTransform(tf.transformations.translation_from_matrix(pred_origin_world_frame),
                            tf.transformations.quaternion_from_matrix(pred_origin_world_frame),
                            depth_data.header.stamp,
                            "pred_origin",
                            "world")

        # Publish pointcloud 0.15s
        # s = time.clock()
        occ_pointcloud_publisher.publish(pred_pointcloud)
        # print('time to publish PC: ', time.clock()- s)

        ############ Only Occupied voxels ###########
        # # Get occupied poses
        # ids_occ = np.argwhere(predict_completion != 0)
        # labels_seg_occ = [predict_completion[id_[0],id_[1],id_[2]] for id_ in ids_occ ]
        # rospy.loginfo("number of predicted occupied voxels:%d", ids_occ.shape[0])
        # poses_occ = ids_occ*param['voxel_unit']*param['downscale']
        
        # # Translate predicted voxels to world coordinate
        # poses_occ_list = poses_occ.tolist()  
        # poses_occ_world_frame = [pred_origin_world_frame@(pose +[1]) for pose in poses_occ_list]
        # poses_occ_world_frame = np.array(poses_occ_world_frame)[:,:3]

        # # Declaring pointcloud
        # occ_pointcloud = PointCloud()

        # # Filling pointcloud header
        # header = Header()
        # header.stamp = depth_data.header.stamp
        # header.frame_id = '/world'
        # occ_pointcloud.header = header

        # channel_r = ChannelFloat32()
        # channel_r.name = "r"
        # channel_g = ChannelFloat32()
        # channel_g.name = "g"
        # channel_b = ChannelFloat32()
        # channel_b.name = "b"

        # # Filling points
        # for pose, label in zip(poses_occ_world_frame,labels_seg_occ):
        #     occ_pointcloud.points.append(Point32(pose[0], pose[1], pose[2]))
        #     channel_r.values.append(colorMap[label][0])
        #     channel_g.values.append(colorMap[label][1])
        #     channel_b.values.append(colorMap[label][2])
        # occ_pointcloud.channels.append(channel_r)
        # occ_pointcloud.channels.append(channel_g)
        # occ_pointcloud.channels.append(channel_b)
        
        # # Publish pointcloud (faster only occupied voxels)
        # occ_pointcloud_publisher.publish(occ_pointcloud)
        
        if save_data:
            my_manhattan = dict()
            my_manhattan['cam_pose'] = cam_pose
            my_manhattan['vox_origin'] = vox_origin
            with open(save_path + '/depth_' + str(counter)+'.pickle', 'wb') as f:
                pickle.dump(depth, f)
            with open(save_path + '/depth_' + str(counter)+'.pickle', 'wb') as f:
                pickle.dump(depth, f)
            with open(save_path + '/rgb_' + str(counter)+ '.pickle', 'wb') as f:
               pickle.dump(rgb, f)
            with open(save_path + '/position_' + str(counter)+ '.pickle', 'wb') as f:
               pickle.dump(position, f)
            with open(save_path + '/my_manhattan_' + str(counter)+ '.pickle', 'wb') as f:
               pickle.dump(my_manhattan, f)

    else: 
        rospy.logerr('Cannot estimate normals')

    if matlab: 
        ##### MATLAB POSITION ESTIMATION #####
        depth_matlab = matlab.double(np.squeeze(depth))
        vox_origin_matlab, cam_pose_matlab = matlab_eng.GetCamPose(depth_matlab,nargout=2)
        print(vox_origin_matlab)
        print(cam_pose_matlab)
        vox_origin_matlab = np.squeeze(np.float32(vox_origin_matlab))
        cam_pose_matlab = np.transpose(np.float32(cam_pose_matlab))

        pred_origin_to_camera = cam_pose_matlab
        pred_origin_to_camera[0][3] -= vox_origin_matlab[0]
        pred_origin_to_camera[1][3] -= vox_origin_matlab[1]
        pred_origin_to_camera[2][3] -= vox_origin_matlab[2]
        pred_origin_to_camera = np.linalg.inv(pred_origin_to_camera)
        
        quat = tf.transformations.quaternion_from_matrix(pred_origin_to_camera)
        quat = quat/ np.linalg.norm(quat)

        tf_broadcaster.sendTransform(tf.transformations.translation_from_matrix(pred_origin_to_camera),
                            quat,
                            rospy.Time.now(),
                            "matlab_pred_origin",
                            "camera")
        position_matlab = depth2position(depth, cam_pose_matlab, vox_origin_matlab)

        if save_data:
            my_manhattan = dict()
            my_manhattan['cam_pose'] = cam_pose_matlab
            my_manhattan['vox_origin'] = vox_origin_matlab
            with open(save_path + '/position_matlab_' + str(counter)+ '.pickle', 'wb') as f:
                pickle.dump(position_matlab, f)
            with open(save_path + '/matlab_manhattan_' + str(counter)+ '.pickle', 'wb') as f:
               pickle.dump(my_manhattan, f)

    counter += 1



if __name__ == '__main__':
    
    #Node init
    rospy.init_node(node_name, anonymous=True)
    print(sys.version)

    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled=True

    #Set ROS params
    rospy.set_param(node_name + '/voxel_size_x', str(param['voxel_size'][0]))
    rospy.set_param(node_name + '/voxel_size_y', str(param['voxel_size'][1]))
    rospy.set_param(node_name + '/voxel_size_z', str(param['voxel_size'][2]))
    rospy.set_param(node_name + '/voxel_unit', str(param['voxel_unit']))
    rospy.set_param(node_name + '/downscale', str(param['downscale']))

    #Start matlab engine
    if matlab: 
        matlab_eng = matlab.engine.start_matlab()
        matlab_eng.addpath('/home/arno/sscnet/matlab_code',nargout=0)
        matlab_eng.addpath('/home/arno/sscnet/matlab_code/utils',nargout=0)
        matlab_eng.addpath('/home/arno/sscnet/demo',nargout=0)
    
    #Declare pointclouds publishers 
    occ_pointcloud_publisher = rospy.Publisher("/ssc/occ_prediction", PointCloud, queue_size=10)
    pose_publisher = rospy.Publisher("/ssc/pred_origin", PoseStamped, queue_size=10)
    # free_pointcloud_publisher = rospy.Publisher("/ssc/free_prediction", PointCloud, queue_size=10)

    #Check CUDA
    if torch.cuda.is_available():
        rospy.loginfo("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())
    else:
        rospy.logerr("Sorry, You DO NOT have a CUDA device!")

    #Declare DDRNet model and load weights
    net = make_model('ddrnet', num_classes=12).cuda()
    load_checkpoint = torch.load(weight_file)

    state_dict = dict()
    for key in load_checkpoint['state_dict'].keys():
        new_key = key.replace('module.','')
        state_dict[new_key] = load_checkpoint['state_dict'][key]
    net.load_state_dict(state_dict)
    net.eval()

    # Declare a TF transform broadcaster
    tf_broadcaster = tf.TransformBroadcaster()

    depth_sub = message_filters.Subscriber(depth_image_topic, Image)
    rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
    pointcloud_sub = message_filters.Subscriber(pointcloud_topic, PointCloud2)
    camera_pose_sub = message_filters.Subscriber(camera_pose_topic, PoseStamped)

    ts = message_filters.TimeSynchronizer([depth_sub, rgb_sub,pointcloud_sub, camera_pose_sub], 1)
    ts.registerCallback(callback)

    print('SSC READY!')
    rospy.spin()
