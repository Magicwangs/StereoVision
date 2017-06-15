#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:11:10 2017

@author: magic
"""
import cv2
import time
from os import listdir
import numpy as np
import FASTCONFIG
import tensorflow as tf
from numba import jit,vectorize,cuda
from FastNet import FastNet
from Post_Process import Post_Process

@jit(nopython=True)
def speed(mat):
    std = np.std(mat)
    mean = np.mean(mat)
    output = np.divide(mat-mean,std)
    return output      

@jit(nopython=True)
def slicespeed(imgRefer, imgTarget, x, y, num):
    referPatch = imgRefer[x-4:x+4+1, y+num-4:y+num+4+1]
    targetPatch = imgTarget[x-4:x+4+1, y+num-4:y+num+4+1]
    return referPatch, targetPatch
    
class StereoMatch:
    def __str__(self):
	return 'Stereo Patch Match'
     
    __repr__ = __str__
    
    def __init__(self,referDir, targetDir, final_dispDir, left_dispDir, right_dispDir, netDir):
        self.referDir = referDir
        self.targetDir = targetDir
        self.final_dispDir = final_dispDir
        self.left_dispDir = left_dispDir
        self.right_dispDir = right_dispDir
        self.netDir = netDir

    def preprocess(self, mat):
        mat = mat.reshape((1, -1))
#        output = speed(mat)
#        output = np.nan_to_num(output)
        return mat
    
    def fast_net_eval(self, imgRefer, imgTarget, y_range, sess, netout, subNet_1, subNet_2, batchLeft, batchRight):        
        self.picWidth = imgRefer.shape[0]
        self.picLen = imgRefer.shape[1]
        disp = np.zeros(imgRefer.shape, dtype=np.float32)
        d_out = np.zeros(imgRefer.shape, dtype=np.float32)
        R_disp = np.zeros(imgRefer.shape, dtype=np.float32)
        R_d_out = np.zeros(imgRefer.shape, dtype=np.float32)
        referSubnet = np.zeros((self.picLen-8, 9*9*FASTCONFIG.num_conv_feature_maps))
        targetSubnet = np.zeros((self.picLen-8, 9*9*FASTCONFIG.num_conv_feature_maps))
#        t0 = time.clock()
        for w in range(self.picWidth-8):
            x = w + 4
            #print x
#            print time.clock()-t0
            batchrange = 200
            i_range = (self.picLen-8)/batchrange
            for i in range(i_range+1):
                y = batchrange*i + 4
                num_range = 200
                if i == i_range:
                    num_range = self.picLen - 8 - batchrange*i
                batch_in_Left = np.empty((num_range, 9, 9))
                batch_in_Right = np.empty((num_range, 9, 9))
                for num in range(num_range):
                    referPatch, targetPatch = slicespeed(imgRefer, imgTarget, x, y, num)
                    batch_in_Left[num] = referPatch
                    batch_in_Right[num] = targetPatch
                subout_1, subout_2 = sess.run([subNet_1,subNet_2], feed_dict = {batchLeft: batch_in_Left, batchRight: batch_in_Right})
                referSubnet[batchrange*i:batchrange*i+num_range] = subout_1[:]
                targetSubnet[batchrange*i:batchrange*i+num_range] = subout_2[:]
#            print time.clock()-t0
            for le in range(self.picLen-8):
                y = le + 4
                if y-y_range > 4:
                    batchsize = y_range+1
                else:
                    batchsize = y-4+1
                referSubnetBatch = np.empty((batchsize, 9*9*FASTCONFIG.num_conv_feature_maps), dtype=np.float32)
                targetSubnetBatch = np.empty((batchsize, 9*9*FASTCONFIG.num_conv_feature_maps), dtype=np.float32)
                for b in range(batchsize):
                    referSubnetBatch[b] = referSubnet[le]
                    targetSubnetBatch[b] = targetSubnet[le-b]
#                referSubnetBatch = np.tile(referSubnet[le], (batchsize,1))
#                targetSubnetBatch = targetSubnet[le-batchsize+1:le+1]
#                targetSubnetBatch = targetSubnetBatch[::-1]
                out = sess.run(netout,feed_dict = {subNet_1: referSubnetBatch, subNet_2: targetSubnetBatch})
#                out = np.zeros((5,1))
                Disp = out.argmax()
                maxVal = np.max(out)
                if maxVal <= 0.5:
                    Disp = 0
                disp[x,y] = Disp
                d_out[x,y] = maxVal
#                print "disp: " + str(Disp) + " maxout: "+ str(maxVal)
            if right_dispDir is not None:
                for le in range(self.picLen-8):
                    y = le + 4
                    if y+y_range < self.picLen-1-4-1:
                        batchsize = y_range+1
                    else:
                        batchsize = self.picLen-4-1 - y+1
                    referSubnetBatch = np.empty((batchsize, 9*9*FASTCONFIG.num_conv_feature_maps), dtype=np.float32)
                    targetSubnetBatch = np.empty((batchsize, 9*9*FASTCONFIG.num_conv_feature_maps), dtype=np.float32)
                    for b in range(batchsize):
                        referSubnetBatch[b] = targetSubnet[le]
                        targetSubnetBatch[b] = referSubnet[le+b]
                    out = sess.run(netout,feed_dict = {subNet_1: referSubnetBatch, subNet_2: targetSubnetBatch})
                    Disp = out.argmax()
                    maxVal = np.max(out)
                    if maxVal <= 0.5:
                        Disp = 0
                    R_disp[x,y] = Disp    
                    R_d_out[x,y] = maxVal
        return disp, d_out, R_disp, R_d_out
    
    def recover(self, sample_N, finalDisp, realWidth, realLen, newWidth, newLen):
        finalDisp = finalDisp*sample_N
        for i in range(sample_N/2):    
            finalDisp = cv2.pyrUp(finalDisp)
        realDisp = np.zeros((realWidth, realLen))
        realDisp[realWidth-newWidth:, realLen-newLen:] = finalDisp[:]
        return realDisp
        
    def stereomatch(self):
        stereo = FastNet(FASTCONFIG.epochs, FASTCONFIG.iteration,
                         FASTCONFIG.num_conv_feature_maps, FASTCONFIG.num_fc_units,
                         FASTCONFIG.learning_rate, FASTCONFIG.momentum,
                         FASTCONFIG.weight_decay, FASTCONFIG.save_Dir)
        sess, netout, subNet_1, subNet_2, batchLeft, batchRight = stereo.netbuild()
        
        myPost_Process = Post_Process()
        
        sample_N = 2
        y_range = 220/sample_N
        picList = listdir(self.referDir)
        picList = ['a_10.bmp','000193_10.png']
        errorList = np.zeros((200, 1))
        disp_err = 0.0
        tau = 3
        picNum = 0
        
        for pic in picList:
            if pic.split('_')[1]=='10.png':
                picNum += 1
                t0 = time.clock()
                imgRefer = cv2.imread(self.referDir + pic, flags=cv2.IMREAD_ANYDEPTH)
                #imgRefer = imgRefer[200:300, 0:300]
                imgTarget = cv2.imread(self.targetDir + pic, flags=cv2.IMREAD_ANYDEPTH)
                #imgTarget = imgTarget[200:300, 0:300]
                realWidth = imgRefer.shape[0]
                realLen = imgRefer.shape[1]
                ## sample slice
                newLen = sample_N*(realLen/sample_N)
                newWidth = sample_N*(realWidth/sample_N)
                imgRefer = imgRefer[realWidth-newWidth:, realLen-newLen:]
                imgTarget = imgTarget[realWidth-newWidth:, realLen-newLen:]
                ## downsample
                for i in range(sample_N/2):
                    imgRefer = cv2.pyrDown(imgRefer)
                    imgTarget = cv2.pyrDown(imgTarget)
                ## net eval
                disp, d_out, R_disp, R_d_out = self.fast_net_eval(imgRefer, imgTarget, y_range, sess, netout, subNet_1, subNet_2, batchLeft, batchRight)
                t1 =  time.clock()-t0
                t0 = time.clock()
                
#                ## recover
#                disp = self.recover(sample_N, disp, realWidth, realLen, newWidth, newLen)
#                R_disp = self.recover(sample_N, R_disp, realWidth, realLen, newWidth, newLen)
#                
                ## left_right_check
                if right_dispDir is not None:
                    finalDisp = myPost_Process.left_right_check(LeftDisp=disp, RightDisp=R_disp)
                else:
                    finalDisp = disp
                t2 = time.clock()-t0
                ## filter
                finalDisp = myPost_Process.twofilter(finalDisp)
                
                ## upsample
                finalDisp = self.recover(sample_N, finalDisp, realWidth, realLen, newWidth, newLen)
                ## eval
                realDisp = np.where(finalDisp == 0, -1, finalDisp)     
            
#            realDisp = realDisp[100:, :]
                preDir = FASTCONFIG.train_Data
                referDispName = preDir + '/disp_noc/' + pic
                originalDisp = cv2.imread(referDispName, flags=cv2.IMREAD_ANYDEPTH)
                referDisp = originalDisp.astype(np.float32)/256.0
                referDisp = np.where(originalDisp == 0, -1, referDisp)
                
                referDisp = np.around(referDisp)
                
        #            referDisp = referDisp[100:, :]
                
                absDisp = abs(referDisp - realDisp)
                absDisp = np.where(referDisp <= 0, 0, absDisp)
                index1 = np.where(absDisp > tau)
                index2 = np.where(referDisp > 0)
                error = len(index1[1])/float(len(index2[1]))
                disp_err = disp_err + error
                errorList[picNum-1] = error
                ## print
                print "======TIME=========="
                print pic
                print "net eval time: " + str(t1/60) + "minutes"
                print "left right check time: " + str(t2) + "seconds"
                print errorList[0:7]
                print "disp_err: " + str(disp_err/picNum)
                print "======TIME=========="
                ## save
                np.save(self.left_dispDir+ pic.split('.')[0] + ".npy", disp)
                np.save(self.left_dispDir + pic.split('.')[0] + "_maxVal.npy", d_out)
                if right_dispDir is not None:
                    np.save(self.right_dispDir+ pic.split('.')[0] + ".npy", R_disp)
                    np.save(self.right_dispDir + pic.split('.')[0] + "_maxVal.npy", R_d_out)
                    
                np.save(self.final_dispDir+ pic.split('.')[0] + ".npy", finalDisp)
                        
if __name__=="__main__":
    picDir = FASTCONFIG.train_Data
    referDir = picDir + "/image_0/"
    targetDir = picDir + "/image_1/"
    final_dispDir = './disp/'
    left_dispDir = "./disp/leftdisp/"
    right_dispDir = None
    right_dispDir = './disp/rightdisp/'
    netDir = "./save/0204"
    match = StereoMatch(referDir, targetDir, final_dispDir, left_dispDir, right_dispDir, netDir)
    out = match.stereomatch()
    
