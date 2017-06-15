#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:18:51 2017

@author: magic
"""

from numba import cuda,jit
import FASTCONFIG
import cv2
import numpy as np
from os import listdir
import time

@jit(nopython=True)
def speed(mat):
    std = np.std(mat)
    mean = np.mean(mat)
    output = np.divide(mat-mean,std)
    return output


#@cuda.jit
@jit(nopython=True)
def getP(D1, D2, direction):
    sgm_P1 = 3.5
    sgm_P2 = 223.0
    sgm_Q1 = 2.0
    sgm_Q2 = 4.0
    sgm_V = 1.75
    sgm_D = 0.02
    if D1 < sgm_D and D2<sgm_D:
        P1 = sgm_P1
        P2 = sgm_P2
    elif D1 >= sgm_D and D2 >= sgm_D:
        P1 = sgm_P1/sgm_Q2
        P2 = sgm_P2/sgm_Q2
    else:
        P1 = sgm_P1/sgm_Q1
        P2 = sgm_P2/sgm_Q1
    if direction >= 2 :
        P1 = P1/sgm_V
    return P1, P2
    
@jit(nopython=True)
def sgm(imgL, imgR, net_cost, originalDisp):
    finalDisp = np.zeros_like(originalDisp, np.float32)
    picWidth = imgL.shape[0]
    picLen = imgL.shape[1]
    d_range = net_cost.shape[2]
    cost_0 = np.zeros_like(net_cost, np.float32)
    cost_1 = np.zeros_like(net_cost, np.float32)
    cost_2 = np.zeros_like(net_cost, np.float32)
    cost_3 = np.zeros_like(net_cost, np.float32)
    for i in range(4):
        if i == 0:
            dx = 0
            dy = 1
#            print(i)
            for w in range(picWidth-8):
                w0 = w+4
                for le in range(picLen-8):
                    le0 = le+4
                    tmp_min = np.min(cost_0[w0-dx, le0-dy])
                    D1 = abs(imgL[w0, le0]-imgL[w0-dx, le0-dy])
                    for d in range(d_range-2):
                        d0 = d+1
                        if le0-d0>0:
                            D2 = abs(imgR[w0, le0-d0]-imgR[w0-dx, le0-d0-dy])
                        else:
                            D2 = 0
                        P1,P2 = getP(D1, D2, 0)
                        min_0 = cost_0[w0-dx, le0-dy, d0]
                        min_1 = cost_0[w0-dx, le0-dy, d0-1]+P1
                        min_2 = cost_0[w0-dx, le0-dy, d0+1]+P1
                        min_3 = tmp_min+P2
                        cost_0[w0, le0, d0] = net_cost[w0, le0, d0]-tmp_min+min(min_0, min_1, min_2, min_3)
        elif i == 1:
            dx = 0
            dy = -1
#            print(i)
            for w in range(picWidth-8):
                w0 = w+4
                for le in range(picLen-8):
                    le0 = picLen-le-4
                    tmp_min = np.min(cost_0[w0-dx, le0-dy])
                    D1 = abs(imgL[w0, le0]-imgL[w0-dx, le0-dy])
                    for d in range(d_range-2):
                        d0 = d+1
                        if le0-d0>0:
                            D2 = abs(imgR[w0, le0-d0]-imgR[w0-dx, le0-d0-dy])
                        else:
                            D2 = 0
                        P1,P2 = getP(D1, D2, 1)
                        min_0 = cost_1[w0-dx, le0-dy, d0]
                        min_1 = cost_1[w0-dx, le0-dy, d0-1]+P1
                        min_2 = cost_1[w0-dx, le0-dy, d0+1]+P1
                        min_3 = tmp_min+P2
                        cost_1[w0, le0, d0] = net_cost[w0, le0, d0]-tmp_min+min(min_0, min_1, min_2, min_3)
        elif i == 2:
            dx = 1
            dy = 0
#            print(i)
            for le in range(picLen-8):
                le0 = le+4
                for w in range(picWidth-8):
                    w0 = w+4
                    tmp_min = np.min(cost_2[w0-dx, le0-dy])
                    D1 = abs(imgL[w0, le0]-imgL[w0-dx, le0-dy])
                    for d in range(d_range-2):
                        d0 = d+1
                        if le0-d0>0:
                            D2 = abs(imgR[w0, le0-d0]-imgR[w0-dx, le0-d0-dy])
                        else:
                            D2 = 0
                        P1,P2 = getP(D1, D2, 2)
                        min_0 = cost_2[w0-dx, le0-dy, d0]
                        min_1 = cost_2[w0-dx, le0-dy, d0-1]+P1
                        min_2 = cost_2[w0-dx, le0-dy, d0+1]+P1
                        min_3 = tmp_min+P2
                        cost_2[w0, le0, d0] = net_cost[w0, le0, d0]-tmp_min+min(min_0, min_1, min_2, min_3)
        else:
            dx = -1
            dy = 0
#            print(i)
            for le in range(picLen-8):
                le0 = le+4
                for w in range(picWidth-8):
                    w0 = picWidth-w-4
                    tmp_min = np.min(cost_3[w0-dx, le0-dy])
                    D1 = abs(imgL[w0, le0]-imgL[w0-dx, le0-dy])
                    for d in range(d_range-2):
                        d0 = d+1
                        if le0-d0>0:
                            D2 = abs(imgR[w0, le0-d0]-imgR[w0-dx, le0-d0-dy])
                        else:
                            D2 = 0
                        P1,P2 = getP(D1, D2, 3)
                        min_0 = cost_3[w0-dx, le0-dy, d0]
                        min_1 = cost_3[w0-dx, le0-dy, d0-1]+P1
                        min_2 = cost_3[w0-dx, le0-dy, d0+1]+P1
                        min_3 = tmp_min+P2
                        cost_3[w0, le0, d0] = net_cost[w0, le0, d0]-tmp_min+min(min_0, min_1, min_2, min_3)
    finalCost = (0.25)*(cost_0+cost_1+cost_2+cost_3)
    for w in range(picWidth-8):
        w0 = w+4
        for le in range(picLen-8):
            le0 = le+4
            originalD = int(originalDisp[w0, le0])
            if originalD-1>=0:
                sgm_0 = finalCost[w0, le0, originalD-1]
            else:
                sgm_0 = 0
            sgm_1 = finalCost[w0, le0, originalD]
            if originalD+1<d_range:
                sgm_2 = finalCost[w0, le0, originalD+1]
            else:
                sgm_2=0
            down = 2*(sgm_2-2*sgm_1+sgm_0)
            if down==0:
                down = 1
            finalDisp[w0, le0] = originalD-(sgm_2-sgm_0)/down
    return finalDisp
        
#@cuda.jit
#def sgm(net_cost, resultDisp, originalDisp, leftPic, rightPic):
#    net_x, net_y, net_c = cuda.grid(3)
#    o_x, o_y = cuda.grid(2)
#    r_x, r_y = cuda.grid(2)
#    left_x, left_y = cuda.grid(2)
#    right_x, right_y = cuda.grid(2)
#    picWidth = net_cost.shape[0]
#    picLen = net_cost.shape[1]
#    direction = 1
#    for d in direction:
#        if d == 1:
#            for w in picWidth:

if __name__=="__main__":
#    preDir = '/home/magic/StereoVision/Code/KITTI_Data/KITTI2012/training/'
    preDir = FASTCONFIG.train_Data
    resultDir = './disp/LR'
    picDir = "./disp/leftdisp"
    tau = 3##误差阈值
    picNum = 0
    leftDir = preDir + '/image_0/'
    rightDir = preDir + '/image_1/'
    picList = listdir(picDir)
    disp_err = 0.0
    
#    picList = ['000000_10.png', '000193_.png']
    #picList = ['000000_10.png','000001_10.png','000002_10.png','000191_10.png','000192_10.png','000193_10.png']
    errorList = np.zeros((200, 1))
    
    for pic in picList:
        if pic.split('_')[1]=='10.png':
            picNum += 1
            imgL = cv2.imread(leftDir + pic, flags=cv2.IMREAD_ANYDEPTH)
            imgR = cv2.imread(rightDir + pic, flags=cv2.IMREAD_ANYDEPTH)
            dispName = pic.split('.')[0]
            
            imgL = speed(imgL)
            imgR = speed(imgR)
            firstDisp = np.load(resultDir + dispName + ".npy")
            net_cost = np.load("./cost/"+ pic.split('.')[0] + ".npy")
#            net_cost = -net_cost
            realDisp = firstDisp
#            realDisp = np.zeros_like(firstDisp)
            t0 = time.clock()
            realDisp = sgm(imgL, imgR, net_cost, firstDisp)
            realDisp=cv2.medianBlur(realDisp, 5)
#            realDisp = cv2.bilateralFilter(realDisp, 5, 7.74,7.74)
            print time.clock()-t0
#            np.save("a.npy", realDisp)
            realDisp = np.where(realDisp == 0, -1, realDisp)     
            
            referDispName = preDir + '/disp_noc/' + pic
            originalDisp = cv2.imread(referDispName, flags=cv2.IMREAD_ANYDEPTH)
            referDisp = originalDisp.astype(np.float32)/256.0
            referDisp = np.where(originalDisp == 0, -1, referDisp)
            
#            referDisp = np.around(referDisp)
            
            absDisp = abs(referDisp - realDisp)
            absDisp = np.where(referDisp <= 0, 0, absDisp)
            index1 = np.where(absDisp > tau)
            index2 = np.where(referDisp > 0)
            error = len(index1[1])/float(len(index2[1]))
            disp_err = disp_err + error
            errorList[picNum-1] = error
    #print errorList[:6]
    disp_err = disp_err/picNum    
    print 'disp_error:'+str(disp_err*100)+'%'
