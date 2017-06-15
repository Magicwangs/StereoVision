#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:08:49 2017

@author: magic
"""
import cv2
import numpy as np
from numba import jit


d_range = 30

@jit(nopython=True)
def init(LeftDisp, RightDisp, MarkMap, picWidth, picLen):
    for w in range(picWidth):
        for le in range(picLen):
            left_d = LeftDisp[w, le]
            right_d = RightDisp[w, np.int64(le-left_d)]
            if abs(left_d - right_d) <= 1:
                MarkMap[w , le] = 1
            else:
                for d in range(d_range):
                    if le - d >= 0:
                        tmp_d = RightDisp[w, np.int64(le-d)]
                        if abs(left_d - tmp_d) <= 1:
                            MarkMap[w, le] = 2
    return MarkMap
  
@jit(nopython=True)
def occulsion(MarkMap, realDisp):
    occIndex = np.where(MarkMap == 0)
    occ_W = occIndex[0]
    occ_Len = occIndex[1]
    for i in range(len(occ_W)):
        w = occ_W[i]
        le = occ_Len[i]
        while True:
            le -= 1
            if le < 0:
                break
            if MarkMap[w, le] == 1:
                realDisp[w, occ_Len[i]] = realDisp[w, le]
                break
    return realDisp
    
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
def SGM(imgL, imgR, net_cost, originalDisp):
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

@jit(nopython=True)    
def median(lst):
    lst=np.sort(lst)
    if len(lst)%2==1:
        return lst[len(lst)//2]
    else:
        return  (lst[len(lst)//2-1]+lst[len(lst)//2])/2.0

@jit(nopython=True)    
def speed(MarkMap, realDisp, mismatchIndex, picLen, picWidth):
    mis_W = mismatchIndex[0]
    mis_Len = mismatchIndex[1]
    for i in range(len(mis_W)):
        d_list = np.zeros(16)
        w = mis_W[i]
        le = mis_Len[i]
        for direct in range(16):
            if direct == 0:
                while True:
                    le -= 1
                    if le < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[0] = realDisp[w, le]
                        break
            if direct == 1:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    w -= 1
                    if w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[1] = realDisp[w, le]
                        break
            if direct == 2:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 1
                    if le > picLen-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[2] = realDisp[w, le]
                        break
            if direct == 3:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    w += 1
                    if w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[3] = realDisp[w, le]
                        break
            if direct == 4:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 1
                    w += 1
                    if le > picLen-1 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[4] = realDisp[w, le]
                        break
            if direct == 5:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 1
                    w -= 1
                    if le > picLen-1 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[5] = realDisp[w, le]
                        break
            if direct == 6:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 1
                    w -= 1
                    if le <0 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[6] = realDisp[w, le]
                        break
            if direct == 7:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 1
                    w += 1
                    if le < 0 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[7] = realDisp[w, le]
                        break
            if direct == 8:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 2
                    w += 1
                    if le > picLen-1 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[8] = realDisp[w, le]
                        break
            if direct == 9:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 2
                    w -= 1
                    if le > picLen-1 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[9] = realDisp[w, le]
                        break
            if direct == 10:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 2
                    w -= 1
                    if le <0 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[10] = realDisp[w, le]
                        break
            if direct == 11:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 2
                    w += 1
                    if le < 0 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[11] = realDisp[w, le]
                        break
            if direct == 12:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 1
                    w += 2
                    if le > picLen-1 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[12] = realDisp[w, le]
                        break
            if direct == 13:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le += 1
                    w -= 2
                    if le > picLen-1 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[13] = realDisp[w, le]
                        break
            if direct == 14:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 1
                    w -= 2
                    if le <0 or w < 0:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[14] = realDisp[w, le]
                        break
            if direct == 15:
                w = mis_W[i]
                le = mis_Len[i]
                while True:
                    le -= 1
                    w += 2
                    if le < 0 or w > picWidth-1:
                        break
                    if MarkMap[w, le] == 1:
                        d_list[15] = realDisp[w, le]
                        break
        realDisp[mis_W[i], mis_Len[i]] = median(d_list)       
    return realDisp

class Post_Process:
    def __str__(self):
        return 'Post_Process of Stereo Version'
        
    __repr__ = __str__
        
    def __init__(self):
        print 'Post_Process Init!'
      
    def left_right_check(self, LeftDisp, RightDisp):
#        dispName = pic.split('.')[0]
#        LeftDisp = np.load(self.resultDir + '/leftdisp/' + dispName + ".npy")
#        RightDisp = np.load(self.resultDir + '/rightdisp/' + dispName + ".npy")
        MarkMap = np.zeros(shape=LeftDisp.shape)
        picWidth = LeftDisp.shape[0]
        picLen = LeftDisp.shape[1]
        ## 默认遮挡 0 ，匹配 1，不匹配 2
        MarkMap = init(LeftDisp, RightDisp, MarkMap, picWidth, picLen)
        ## correct                        
        realDisp = np.where(MarkMap == 1, LeftDisp, 0)
        ## occlusion
        realDisp = occulsion(MarkMap, realDisp)
        ## mismatch      
        mismatchIndex = np.where(MarkMap == 2)
        realDisp = speed(MarkMap, realDisp, mismatchIndex, picLen, picWidth)
        return realDisp
            
    def twofilter(self, disp):
        disp = disp.astype(np.float32)
        realDisp =  cv2.medianBlur(disp, 5)
        realDisp = cv2.bilateralFilter(realDisp,15,4.6,4.6)
        return realDisp
            
if __name__=="__main__":
    pass
            
            
            
