#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy
from std_msgs.msg import String, Header
from trajectory_msgs.msg import MultiDOFJointTrajectory
from sensor_msgs.msg import PointCloud, ChannelFloat32, Image
from geometry_msgs.msg import Point32, Pose
import tf
import message_filters


import sys
sys.path.append("../SSC")
from dataloaders import NYUDataset
from models import make_model
import numpy as np
import os
import datetime
from tqdm import tqdm
import ros_numpy # package for conversion ros_msg to numpy
import time
import torch 
from torch.autograd import Variable

import matlab.engine
from dataloaders import NYUDataset

depth_image_topic = '/unreal/unreal_sensor_model/ue_depth_image_out'
rgb_image_topic = '/unreal/unreal_sensor_model/ue_color_image_out'
dir_test = "/home/arno/Documents/NYUCAD_single_example"
weight_file = '/home/arno/SSC/weights/cpBest_SSC_DDRNet3.pth.tar'
VOX_UNIT_OUT = 0.08
dir_test = "/home/arno/Documents/NYUCAD_single_example"
dataset=NYUDataset(dir_test, istest=True)

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


def callback(depth_data, rgb_data):
    print('rostime:',rospy.get_time())
    depth = ros_numpy.numpify(depth_data)
    depth_matlab = matlab.double(np.squeeze(depth))
    rgb = ros_numpy.numpify(rgb_data)
    rgb = np.moveaxis(rgb,-1,0)
    #np.save('depth.npy', depth)
    #np.save('rgb.npy', rgb)
    
    print('RGB_SHAPE:', rgb.shape,'DEPTH_SHAPE:',depth.shape)
    assert depth_data.header.stamp == rgb_data.header.stamp

    try:
        (trans,rot) = listener.lookupTransform('/world', '/camera', depth_data.header.stamp)    
        trans_mat = tf.transformations.translation_matrix(trans)
        rot_mat   = tf.transformations.quaternion_matrix(rot)
        cam_pose = np.dot(trans_mat, rot_mat)
        np.save('cam_pose.npy', cam_pose)
        vox_origin = matlab_eng.MYperpareDataTest(depth_matlab)
        position  = dataset._depth2position(depth, cam_pose, vox_origin, dataset.param)
    
    except:
        rospy.logerr('FAIL')
        return
        
    var_depth = Variable(torch.unsqueeze(torch.unsqueeze(torch.tensor(depth),0),0).float()).cuda()
    var_rgb = Variable(torch.unsqueeze(torch.tensor(rgb),0).float()).cuda()
    position = torch.tensor(position).long().cuda()
    print(var_depth.shape,var_rgb.shape,position.shape)
        
    with torch.no_grad():
        pred = net(x_depth = var_depth, x_rgb=var_rgb, p = position)
    torch.cuda.empty_cache()
    y_pred = pred.cpu().data.numpy()
    print(y_pred.shape)
    
    predict_completion = np.squeeze(np.argmax(y_pred, axis=1))
    ids_occ = np.argwhere(predict_completion != 0)
    labels_seg_occ = [predict_completion[id_[0],id_[1],id_[2]] for id_ in ids_occ ]
    rospy.loginfo("number of predicted occupied voxels:%d", ids_occ.shape[0])
    poses_occ = ids_occ*VOX_UNIT_OUT 

    # Declaring pointcloud
    occ_pointcloud = PointCloud()
    
    # Filling pointcloud header
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = '/camera'
    occ_pointcloud.header = header

    channel_r = ChannelFloat32()
    channel_r.name = "r"
    channel_g = ChannelFloat32()
    channel_g.name = "g"
    channel_b = ChannelFloat32()
    channel_b.name = "b"

    # Filling points
    #print('FILLING POINTS')
    for pose, label in zip(poses_occ,labels_seg_occ):
        occ_pointcloud.points.append(Point32(pose[0], pose[1], pose[2]))
        channel_r.values.append(colorMap[label][0])
        channel_g.values.append(colorMap[label][1])
        channel_b.values.append(colorMap[label][2])
    occ_pointcloud.channels.append(channel_r)
    occ_pointcloud.channels.append(channel_g)
    occ_pointcloud.channels.append(channel_b)
    # Publish pointcloud
    pointcloud_publisher.publish(occ_pointcloud)



if __name__ == '__main__':
    
    #Node init
    rospy.init_node('ssc', anonymous=True)
    print(sys.version)
    
    #Start matlab engine
    matlab_eng = matlab.engine.start_matlab()
    matlab_eng.addpath('/home/arno/sscnet/matlab_code',nargout=0)
    matlab_eng.addpath('/home/arno/sscnet/matlab_code/utils',nargout=0)
    matlab_eng.addpath('/home/arno/sscnet/demo',nargout=0)
    
    #Declare pointcloud publisher 
    pointcloud_publisher = rospy.Publisher("/ssc/prediction", PointCloud, queue_size=10)

    #Check CUDA
    if torch.cuda.is_available():
        rospy.loginfo("Great, You have {} CUDA device!".format(torch.cuda.device_count()))
    else:
        rospy.loginfo("Sorry, You DO NOT have a CUDA device!")
    
    #Declare dataset
    dataset=NYUDataset(dir_test, istest=True)

    #Declare DDRNet model and load weights
    net = make_model('ddrnet', num_classes=12).cuda()
    load_checkpoint = torch.load(weight_file)

    state_dict = dict()
    for key in load_checkpoint['state_dict'].keys():
        new_key = key.replace('module.','')
        state_dict[new_key] = load_checkpoint['state_dict'][key]
    net.load_state_dict(state_dict)
    net.eval()

    # Let time to register 
    rospy.sleep(3)

    #Declare a TF transform listener to get camera position
    listener = tf.TransformListener()

    depth_sub = message_filters.Subscriber(depth_image_topic, Image)
    rgb_sub = message_filters.Subscriber(rgb_image_topic, Image)
    
    ts = message_filters.TimeSynchronizer([depth_sub, rgb_sub], 10)
    ts.registerCallback(callback)

    # Let time to register 
    rospy.sleep(3)

    rospy.spin()

    # # Declare PALNet model and load weights    
    # net = models.PALNet().cuda()
    # net.eval()
    # load_checkpoint = torch.load("/home/arno/PALNet/PALNet_weights_NYUCAD.pth.tar")
    # state_dict = dict()
    # for key in load_checkpoint['state_dict_G'].keys():
    #     new_key = key.replace('module.','')
    #     state_dict[new_key] = load_checkpoint['state_dict_G'][key]
    # net.load_state_dict(state_dict)
    
    # while not rospy.is_shutdown():

    #     #try:
    #     rospy.loginfo('TIME:%s',rospy.get_time())
    #     depth_msg = rospy.wait_for_message(depth_image_topic, Image, timeout=None)
    #     rgb_msg = rospy.wait_for_message(rgb_image_topic, Image, timeout=None)
    #     print('HEADER DEPTH',depth_msg.header)
    #     print('HEADER DEPTH',rgb_msg.header)
    #     depth = ros_numpy.numpify(depth_msg)
    #     rgb = ros_numpy.numpify(rgb_msg)

    #     # Lookup camera pose
    #     (trans,rot) = listener.lookupTransform('/world', '/camera', depth_msg.header.stamp)    
    #     trans_mat = tf.transformations.translation_matrix(trans)
    #     rot_mat   = tf.transformations.quaternion_matrix(rot)
    #     cam_pose = np.dot(trans_mat, rot_mat)
    #     #print(cam_pose)

    #     # Transform depth image to TSDF
    #     # binary_vox, _, position = dataset._depth2voxel(depth, cam_pose, vox_origin, unit=VOX_UNIT_IN)
    #     # ftsdf = np.reshape(binary_vox,(1, 1, 240, 144, 240))
    #     # var_3d_ftsdf = Variable(torch.tensor(ftsdf).float()).cuda()

    #     # Predict SSC
    #     # depth = depth.reshape((1,) + depth.shape)
    #     # depth = torch.tensor(depth)
    #     # var_2d_depth = Variable(torch.unsqueeze(depth,0).float()).cuda()
    #     # position = torch.tensor(position).long().cuda()

    #     # with torch.no_grad():
    #     #     pred = net( x_depth=var_2d_depth, p=position)
    #     # torch.cuda.empty_cache()
    #     # pred = pred.cpu().data.numpy()
    #     # rospy.loginfo(pred.shape)
        
        
    #     ####################

    #     # _name = '/home/arno/Documents/NYUCAD_single_example/NYU0001_0000'
    #     # npz_file = np.load(_name + '.npz') 
    #     # voxels = npz_file['voxels']
    #     # ftsdf=voxels.squeeze()
    #     # tsdf_hr = ftsdf 
    #     # tsdf = dataset._downsample_tsdf(tsdf_hr, dataset.downsample)
    #     # # print(npz_file.keys())
    #     # depth = npz_file['depth']
    #     # position = npz_file['position']

    #     # var_2d_depth = Variable(torch.unsqueeze(torch.tensor(depth),0).float()).cuda()
    #     # var_3d_ftsdf = Variable(torch.unsqueeze(torch.unsqueeze(torch.tensor(ftsdf),0),0).float()).cuda()

    #     # position = torch.tensor(position).long().cuda()
    #     # with torch.no_grad():
    #     #     pred = net(x_tsdf=var_3d_ftsdf, x_depth=var_2d_depth, p=position)
    #     #     # pred = net(x_depth=var_2d_depth, p=position)
    #     # torch.cuda.empty_cache()
    #     # pred = pred.cpu().data.numpy()
    #     #####################3

    #     # predict_completion = np.squeeze(np.argmax(pred, axis=1))
    #     # ids_occ = np.argwhere(predict_completion != 0)
    #     # labels_seg_occ = [predict_completion[id_[0],id_[1],id_[2]] for id_ in ids_occ ]
    #     # rospy.loginfo("number of predicted occupied voxels:%d", ids_occ.shape[0])
    #     # poses_occ = ids_occ*VOX_UNIT_OUT 

    #     # # Declaring pointcloud
    #     # occ_pointcloud = PointCloud()
        
    #     # # Filling pointcloud header
    #     # header = Header()
    #     # header.stamp = rospy.Time.now()
    #     # header.frame_id = '/camera'
    #     # occ_pointcloud.header = header

    #     # channel_r = ChannelFloat32()
    #     # channel_r.name = "r"
    #     # channel_g = ChannelFloat32()
    #     # channel_g.name = "g"
    #     # channel_b = ChannelFloat32()
    #     # channel_b.name = "b"

    #     # # Filling points
    #     # #print('FILLING POINTS')
    #     # for pose, label in zip(poses_occ,labels_seg_occ):
    #     #     occ_pointcloud.points.append(Point32(pose[0], pose[1], pose[2]))
    #     #     channel_r.values.append(colorMap[label][0])
    #     #     channel_g.values.append(colorMap[label][1])
    #     #     channel_b.values.append(colorMap[label][2])
    #     # occ_pointcloud.channels.append(channel_r)
    #     # occ_pointcloud.channels.append(channel_g)
    #     # occ_pointcloud.channels.append(channel_b)
    #     # # Publish pointcloud
    #     # pointcloud_publisher.publish(occ_pointcloud)

    #     # except: 
    #     #     rospy.logerr('FAILFAILFAIL')
            
    #     rate.sleep()