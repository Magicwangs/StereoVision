# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:07:48 2016

@author: MagicWang
"""

import cv2
import numpy as np
import configs
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot3D():
    fig = plt.figure()
#    ax=plt.subplot(111,projection='3d')
    x = configs.threeD[:,:,0]
    y = configs.threeD[:,:,1]
    z = configs.threeD[:,:,2]
    xx = np.resize(x, (640*480,))
    yy = np.resize(y, (640*480,))
    zz = np.resize(z, (640*480,))
    xx = np.where(zz < 1000 , 0, xx)
    yy = np.where(zz < 1000 , 0, yy)
    zz = np.where(zz < 1000 , 0, zz)
    xx = np.where(zz > 1500 , 0, xx)
    yy = np.where(zz > 1500 , 0, yy)
    zz = np.where(zz > 1500 , 0, zz)
    plt.plot(xx,yy,".")
    plt.legend()
#    ax.scatter(xx, yy, zz, c='b')
#    ax.legend()
    plt.show()




# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:        
        print configs.threeD[y][x]

## 响应trackerbar改变
def onChangeNum(pos):
    configs.num = pos
    Pic_Match(imgL, imgR)
    
def onChangeBlockSize(pos):
    configs.blockSize = pos
    if configs.blockSize % 2 == 0:
        configs.blockSize += 1
    if configs.blockSize < 5:
        configs.blockSize = 5    
    Pic_Match(imgL, imgR)
    
## 匹配
def Pic_Match(imgL, imgR):
    ## SAD Window numDisparities:范围,最大视差 blockSize:窗的大小
    stereo = cv2.StereoBM_create(numDisparities=16*configs.num, blockSize=configs.blockSize)
    ## semi-global block matching
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*configs.num, blockSize=configs.blockSize)
    ## disparity是带4bit小数，所以实际值应该除以16    
    disparity = stereo.compute(imgL, imgR)
    ## 归一化
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    configs.threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., configs.Q)
    cv2.imshow("depth", disp)


if __name__=="__main__":
    cv2.namedWindow("left")
    cv2.namedWindow("right")
    cv2.namedWindow("depth")
    cv2.moveWindow("left", 0, 0)
    cv2.moveWindow("right", 600, 0) 
    ## 必须读入原始数据
    imgL = cv2.imread('./opencv/tmp_L.png', flags=cv2.IMREAD_ANYDEPTH)
    imgR = cv2.imread('./opencv/tmp_R.png', flags=cv2.IMREAD_ANYDEPTH)
    
#    img1_rectified = cv2.remap(img1, configs.left_map1, configs.left_map2, cv2.INTER_LINEAR)
#    img2_rectified = cv2.remap(img2, configs.right_map1, configs.right_map2, cv2.INTER_LINEAR)
#    #转换成灰度
#    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
#    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    
    cv2.createTrackbar("num", "depth", 1, 10, onChangeNum)
    cv2.createTrackbar("blockSize", "depth", 5, 255, onChangeBlockSize)
    
    cv2.setMouseCallback("depth", callbackFunc, None)
    
    cv2.imshow("left", imgL)
    cv2.imshow("right", imgR)
    
    cv2.setTrackbarPos("num", "depth", configs.num)
    cv2.setTrackbarPos("blockSize", "depth", configs.blockSize)
    
    ## SAD Window numDisparities:范围 窗的大小
    stereo = cv2.StereoBM_create(numDisparities=16*configs.num, blockSize=configs.blockSize)
    disparity = stereo.compute(imgL, imgR)
    
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    configs.threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., configs.Q)
    three = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., configs.Q)
    plot3D()
    cv2.imshow("depth", disp)  
    np.savetxt("3D.txt", disparity.astype(np.float32)/16.)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    