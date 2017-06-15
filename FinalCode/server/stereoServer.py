# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:13:37 2017

@author: MagicWang
"""
import cv2
import random
import time
from os import listdir
import numpy as np
import FASTCONFIG
import tensorflow as tf
from numba import jit,vectorize,cuda
from NewFastNet import FastNet
from Post_Process import Post_Process
from Post_Process import SGM


@jit(nopython=True)
def speed(mat):
    std = np.std(mat)
    mean = np.mean(mat)
    output = np.divide(mat-mean,std)
    return output
    
    
class StereoMatch:
    def __str__(self):
	return 'Stereo Patch Match'
     
    __repr__ = __str__
    
    def __init__(self, referPic, targetPic, netDir, dispPic):
        self.referPic = referPic
        self.targetPic = targetPic
        self.netDir = netDir
        self.dispPic = dispPic
        print "StereoMatch Init"
    
    def fast_net_eval(self, imgRefer, imgTarget, y_range, stereo, LRCheck):
        self.picWidth = imgRefer.shape[0]
        self.picLen = imgRefer.shape[1]   
        disp = np.zeros(imgRefer.shape, dtype=np.float32)
        R_disp = np.zeros(imgRefer.shape, dtype=np.float32)
        net_cost = np.zeros((self.picWidth, self.picLen, y_range), dtype=np.float32)

        sess, NetCost, Net_L_disp, Net_R_disp, picL, picR = stereo.lineNet(y_range, self.picLen)
        t0 = time.clock()
        if LRCheck:
            for w in range(self.picWidth-8):
                x = w + 4
                pic_L_in = imgRefer[w:w+9]
                pic_R_in = imgTarget[w:w+9]
                disp[x, 4: self.picLen-4], R_disp[x, 4: self.picLen-4], net_cost[x, 4: self.picLen-4]= sess.run([Net_L_disp, Net_R_disp, NetCost], feed_dict = {picL: pic_L_in, picR: pic_R_in})
        else:
            for w in range(self.picWidth-8):
                x = w + 4
                pic_L_in = imgRefer[w:w+9]
                pic_R_in = imgTarget[w:w+9]
                disp[x, 4: self.picLen-4], net_cost[x, 4: self.picLen-4]= sess.run([Net_L_disp, NetCost], feed_dict = {picL: pic_L_in, picR: pic_R_in})
        nettime = time.clock()-t0
        sess.close()
        del sess
        return nettime, disp, R_disp, net_cost
        
    def stereomatch(self):
        stereo = FastNet(FASTCONFIG.epochs, FASTCONFIG.iteration,
                         FASTCONFIG.num_conv_feature_maps, FASTCONFIG.num_fc_units,
                         FASTCONFIG.learning_rate, FASTCONFIG.momentum,
                         FASTCONFIG.weight_decay, self.netDir)
        myPost_Process = Post_Process()
        
        y_range = 30
        
        LRCheck = True
        SGMFlag = True
        FilterFlag = True
        
        imgRefer = cv2.imread(self.referPic, flags=cv2.IMREAD_ANYDEPTH)
        imgTarget = cv2.imread(self.targetPic, flags=cv2.IMREAD_ANYDEPTH)
        
        ## net eval
        nettime, disp, R_disp, netcost = self.fast_net_eval(imgRefer, imgTarget,  y_range,stereo, LRCheck)
        ## left_right_check
        t0 = time.clock()
        
        if LRCheck:
            LRDisp = myPost_Process.left_right_check(LeftDisp=disp, RightDisp=R_disp)
        else:
            LRDisp = disp
            
        t2 = time.clock()-t0
        
        imgRefer = speed(imgRefer)
        imgTarget = speed(imgTarget)
        
        ## SGM
        t0 = time.clock()
        if SGMFlag:
            finalDisp = SGM(imgRefer, imgTarget, netcost, LRDisp)
        else:
            finalDisp = LRDisp
        t3 = time.clock()-t0
        ## filter
        if FilterFlag:                   
            print "Filter"
            finalDisp = myPost_Process.twofilter(finalDisp)
        
        ## print
        print "======TIME=========="
        print self.netDir
        print "net eval time: " + str(nettime) + "seconds"
        print "left right check time: " + str(t2) + "seconds"
        print "SGM time: "+str(t3)+"seconds"
        print "======TIME=========="
        
        finalDisp = np.where(finalDisp<0, 0, finalDisp)
        saveDisp = finalDisp*256.0
        saveDisp = np.where(finalDisp == 0, 0, saveDisp)
        saveDisp = np.where(saveDisp < 0, 0, saveDisp)
        saveDisp = np.where(saveDisp > 65535, 0, saveDisp)
        saveDisp = saveDisp.astype(np.uint16)
        ## save
       
        cv2.imwrite(self.dispPic, saveDisp)
                        
if __name__=="__main__":
    
    referPic = "./upload/tmp_L.png"
    targetPic = "./upload/tmp_R.png"
    dispPic = "./upload/disp.png"

    netDir = "./save/0228/9"
    
    match = StereoMatch(referPic, targetPic, netDir, dispPic)
    out = match.stereomatch()
