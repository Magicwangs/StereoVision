# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:27:26 2017

@author: MagicWang
"""
import cv2
import numpy as np

## IntrinsicMa
left_camera_mat = np.array([[1103.0359, 0., 224.5895],
                            [0., 1102.7981, 225.3511],
                            [0., 0., 1.]])


## RadialDistortion and TangentialDistortion
left_distortion = np.array([[-0.1429, 0.3633, 0.0015, -0.0044, 0.]])



right_camera_mat = np.array([[1106.3825, 0, 345.3070],
                            [0., 1107.4833, 269.6711],
                            [0., 0., 1.]])

right_distortion = np.array([[-0.1381, 0.0950, 0.0035, -0.0009, 0.]])

## RotationOfCamera2
#R = np.array([[0.9997, 0.0062, -0.0231],
#              [-0.0054, 0.9995, 0.0326],
#              [0.0233, -0.0325, 0.9992]])

R = np.array([[0.9997, 0.0058, -0.0237],
              [-0.0066, 0.9992, -0.0386],
              [0.0235, 0.0387, 0.9989]])

## TranslationOfCamera2
T = np.array([-33.7196, 0.0530, 0.3148])

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