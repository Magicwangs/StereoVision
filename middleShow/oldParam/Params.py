# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:27:26 2017

@author: MagicWang
"""
import cv2
import numpy as np

## IntrinsicMa
left_camera_mat = np.array([[1123.7632, 0., 243.0600],
                            [0., 1121.3437, 220.1881],
                            [0., 0., 1.]])

## RadialDistortion and TangentialDistortion
left_distortion = np.array([[-0.1573, 0.6994, 0.0048, -0.0020, 0.]])



right_camera_mat = np.array([[1131.3162, 0, 357.7593],
                            [0., 1130.4406, 267.6306],
                            [0., 0., 1.]])

right_distortion = np.array([[-0.1746, 0.8458, 0.0068, 0.0003, 0.]])

## RotationOfCamera2
#R = np.array([[0.9997, 0.0062, -0.0231],
#              [-0.0054, 0.9995, 0.0326],
#              [0.0233, -0.0325, 0.9992]])

R = np.array([[0.9999, 0.0056, -0.0104],
              [-0.0060, 0.9991, -0.0430],
              [0.0101, 0.0430, 0.9990]])

## TranslationOfCamera2
T = np.array([-29.4093, -0.1669, -0.4104])

size = (640, 480)  ##长，宽
newsize = (640, 480)

## Bouguet算法
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_mat, left_distortion,
                                                                  right_camera_mat, right_distortion,
                                                                  size, R, T)
#校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_mat, left_distortion, R1, P1, newsize, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_mat, right_distortion, R2, P2, newsize, cv2.CV_16SC2)

num = 1
blockSize = 10
threeD = None