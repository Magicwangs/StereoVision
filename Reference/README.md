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

作者：姚鹏
链接：https://www.zhihu.com/question/37796523/answer/92309937
来源：知乎
著作权归作者所有，转载请联系作者获得授权。

立体匹配(Stereo Matching)算法主要分为全局(global)和局部(local)算法，其中全局算法步骤是：匹配代价计算(matching cost computation)，视差计算(disparity computation)和视差精化(disparity refinement)；局部算法步骤是：匹配代价计算(matching cost computation)，代价聚合(cost aggregation)，视差计算(disparity computation)和视差精化(disparity refinement)。所有的立体匹配算法只能计算到前2/3步，也就是视差精化之前，得到了初始视差图(initial disparity map)，你会发现它有很多的黑色区域，特别是左视差图的最左边，因为那是遮挡区域(occluded regions)，而立体匹配的算法只能计算出非遮挡区域(non-occluded regions)的视差，所以要将右视差图计算出来并进行左右一致性检查(Left-Regiht Crosscheck)来进行遮挡区域的填充，LeCun的论文虽然是采用了机器学习的方法产生视差图，但是仍然避免不了遮挡区域的处理问题，所以会引入亚像素增强(subpixel enhancement)这个策略(strategy)来进行视差精化(也可以叫后处理-post processing)。

## A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation
一个大型数据集训练卷积网络 for 差异，光流，现场流估计
