# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:25:00 2016

@author: MagicWang
"""
import cv2
import numpy as np

## IntrinsicMa 
left_camera_mat = np.array([[534.2814, 0., 341.0507],
                            [0., 534.1941, 236.2255],
                            [0., 0., 1.]])
                            
## RadialDistortion and TangentialDistortion
left_distortion = np.array([[-0.2915, 0.1143, 0.0009, -0.0003, 0.]])

right_camera_mat = np.array([[537.1409, 0., 325.7141],
                            [0., 536.8004, 251.5440],
                            [0., 0., 1.]])
right_distortion = np.array([[-0.2883, 0.1045, -0.0002, 0.0002, 0.]])

## RotationOfCamera2
R = np.array([[1., 0.0031, 0.0044],
              [-0.0031, 1., -0.0077],
              [-0.0045, 0.0077, 1.]])
              
## TranslationOfCamera2
T = np.array([-83.1137, 0.9955, 0.0604])

size = (640, 480)  ##长，宽

## Bouguet算法
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_mat, left_distortion, 
                                                                  right_camera_mat, right_distortion,
                                                                  size, R, T)
#校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_mat, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_mat, right_distortion, R2, P2, size, cv2.CV_16SC2)

num = 2
blockSize = 28
threeD = None