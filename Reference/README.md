**TABLE OF CONTENTS**		
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [文档学习](#文档学习)
	- [Stereo Vision: Algorithms and Applications](#stereo-vision-algorithms-and-applications)
		- [Introduction](#introduction)
		- [Overview](#overview)
		- [Algorithms](#algorithms)
		- [Computational optimizations](#computational-optimizations)
		- [Hardware implementation](#hardware-implementation)
		- [Applications](#applications)
	- [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](#stereo-matching-by-training-a-convolutional-neural-network-to-compare-image-patches)
	- [A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation](#a-large-dataset-to-train-convolutional-networks-for-disparity-optical-flow-and-scene-flow-estimation)

<!-- /TOC -->
# 文档学习
## Stereo Vision: Algorithms and Applications
立体视觉：算法和应用  
### Introduction
binocular stereo vision systems:双目立体视觉系统  
dense stereo algorithms：密度立体算法  
stereo vision applications：立体视觉应用  
Epipolar constraint：对极约束  
标准模式下（两摄像头平行）：  
disparity = Xr-Xt 视差：距离相机越近的点，disparity越大  
range field（Horopter）两眼视界，disparity range[dmin,dmax]  
![](http://chart.googleapis.com/chart?cht=tx&chl=z=\frac{b*f}{d})

### Overview
概述步骤：				
Calibration (offline)离线标定：焦距、相片中心、透镜畸变，直线不再是直线，获取相机的内参（焦距，图像中心，畸变系数等）和外参（R（旋转）矩阵T（平移）矩阵  
还可以得到每个像素点之间的距离，也可以计算出每一个点的x,y,d的值
————棋盘，摄像头  
Rectification双目矫正：根据这5个变量还原出来真实的图片  
Stereo correspondence匹配：找到对应的点	disparity map视差图，块匹配？  
Triangulation三角测量：(x,y)->(x,y,z),z=b*f/d  



### Algorithms
算法
### Computational optimizations
计算优化
### Hardware implementation
硬件实现
### Applications
应用

## Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches
通过训练比较图像补丁的卷积神经网络来立体视觉匹配

## A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation
一个大型数据集训练卷积网络 for 差异，光流，现场流估计
