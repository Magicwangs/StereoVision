# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:13:05 2016

@author: MagicWang
"""

import cv2
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

flag = 0  ## 0:BM，1:SGBM
preDir = 'E:/KITTI_DataSet/KITTI2012/data_stereo_flow/training/'
resultDir = './opencv_Result/'
tau = 3 ##误差阈值
picNum = 0

if __name__=="__main__":
    leftDir = preDir + 'image_0/'
    rightDir = preDir + 'image_1/'
    picList = listdir(leftDir)
    disp_err = 0.0
    
    num = 6
#    blocksize = 17能达到25.08% 25.233%，19：25.07%，25.19%
    blockSize = 17
    
    picList = ['000000_10.png','000001_10.png']
    
    for pic in picList:
        if pic.split('_')[1]=='10.png':
            picNum += 1
            imgL = cv2.imread(leftDir + pic, 0)
            imgR = cv2.imread(rightDir + pic, 0)
            if flag:  
                ## SAD Window numDisparities:范围,最大视差 blockSize:窗的大小
                stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
            else:     
                ## semi-global block matching
                stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*num, blockSize=blockSize)
            ## disparity是带4bit小数，所以实际值应该除以16    
            disparity = stereo.compute(imgL, imgR)
            realDisp = disparity.astype(np.float32)/16.
            
            referDispName = preDir + 'disp_noc/' + pic
            originalDisp = cv2.imread(referDispName, flags=cv2.IMREAD_ANYDEPTH)
            referDisp = originalDisp.astype(np.float32)/256.0
            referDisp = np.where(originalDisp == 0, -1, referDisp)
            
            absDisp = abs(referDisp - realDisp)
            absDisp = np.where(referDisp <= 0, 0, absDisp)
            index1 = np.where(absDisp > tau)
            index2 = np.where(referDisp > 0)
            disp_err = disp_err + len(index1[1])/float(len(index2[1]))
            
#            txtName = resultDir + pic.split('.')[0] + '.txt'
#            np.savetxt(txtName, realDisp)
    ## 归一化
    disp_1 = cv2.normalize(realDisp, realDisp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_2 = cv2.normalize(referDisp, referDisp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_err = disp_err/picNum
    
    plt.figure(num='disp_error')
    plt.suptitle('disp_error:'+str(disp_err*100)+'%', fontsize=18)
    plt.subplot(2,1,1)    
    plt.imshow(disp_1)
    plt.title('opencv_disp', fontsize=14)
    plt.subplot(2,1,2)    
    plt.imshow(disp_2)
    plt.title('refer_disp', fontsize=14)

             

