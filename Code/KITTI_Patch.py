# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 20:03:52 2016

@author: MagicWang
"""
import cv2
import numpy as np
from os import listdir
import random
import os

class KITTI_Patch:
    def __str__(self):
	return 'KITTI_Patch'

    __repr__ = __str__

    def __init__(self, preDir, resultDir,
                 dataset_neg_low, dataset_neg_high,
                 dataset_pos, input_patch_size,
                 batchSize):
        self.preDir = preDir
        self.resultDir = resultDir
        self.dataset_neg_low = dataset_neg_low
        self.dataset_neg_high = dataset_neg_high
        self.dataset_pos = dataset_pos
        self.input_patch_size = input_patch_size
        self.inter_len = input_patch_size[0]/2
        self.leftDir = preDir + 'image_0/'
        self.rightDir = preDir + 'image_1/'
        self.dispDir = preDir + 'disp_noc/'
        self.Dir_1 = resultDir + 'Batch_1/'
        self.Dir_2 = resultDir + 'Batch_2/'
        self.flagDir = resultDir + 'Flag/'
        self.picWidth = 370
        self.picLen = 1226
        self.batchSize = batchSize
        self.nextBatchNum = 0

        self.check()
        print 'KITTI Patch Init'

    def check(self):
        try:
            if not os.path.exists(self.Dir_1):
                os.mkdir(self.Dir_1)
            if not os.path.exists(self.Dir_2):
                os.mkdir(self.Dir_2)
            if not os.path.exists(self.flagDir):
                os.mkdir(self.flagDir)
        except:
            print ("Failed to create directory:")
            exit()

    def preprocess(self, mat):
        mat = mat.reshape((1, -1))
        std = np.std(mat)
        mean = np.mean(mat)
        output = np.divide(mat-mean,std)
        output = np.nan_to_num(output)
        return output

    def indexNumInit(self):
        self.indexNumList = np.arange(0, 128)

    def nextIndex(self):
        numSize = self.indexNumList.size
        random_i = random.randrange(0, numSize)
        nextNum_1 = self.indexNumList[random_i]
        self.indexNumList = np.delete(self.indexNumList, random_i)
        numSize = numSize - 1
        random_i = random.randrange(0, numSize)
        nextNum_2 = self.indexNumList[random_i]
        self.indexNumList = np.delete(self.indexNumList, random_i)
        return nextNum_1, nextNum_2

    def create_batch(self):
        batchNum = 0
        patchNum = 0
        Batch_1 = None
        Batch_2 = None
        flag = np.zeros((1, 9*9), dtype=np.float32)
        picList = listdir(self.leftDir)
        picList = ['000000_10.png','000001_10.png']
        for pic in picList:
            if pic.split('_')[1]=='10.png':
                imgL = cv2.imread(self.leftDir + pic, flags=cv2.IMREAD_ANYDEPTH)
                imgR = cv2.imread(self.rightDir + pic, flags=cv2.IMREAD_ANYDEPTH)

                originalDisp = cv2.imread(self.dispDir + pic, flags=cv2.IMREAD_ANYDEPTH)
                referDisp = originalDisp.astype(np.float32)/256.0
    #            referDisp = np.where(originalDisp == 0, -1, referDisp)
                index = np.where(referDisp > 0)
                for i in range(len(index[0])):
                    pos_X = index[0][i]
                    pos_Y = index[1][i]
                    if pos_X-self.inter_len > 0 and pos_X+self.inter_len+1 < self.picWidth:
                        if pos_Y-self.inter_len > 0 and pos_Y+self.inter_len+1 < self.picLen:
                            d = referDisp[pos_X][pos_Y]
                            Opos = random.uniform(-self.dataset_pos, self.dataset_pos)
                            pos_Patch_1 = imgL[pos_X-self.inter_len:pos_X+self.inter_len+1, pos_Y-self.inter_len:pos_Y+self.inter_len+1]
                            pos_R_Y = int(pos_Y - d + Opos)
                            if pos_R_Y-self.inter_len < 0 or pos_R_Y+self.inter_len+1 >= self.picLen:
                                continue
                            pos_Patch_2 = imgR[pos_X-self.inter_len:pos_X+self.inter_len+1, pos_R_Y-self.inter_len:pos_R_Y+self.inter_len+1]

                            if random.randint(0,1):
                                Oneg = random.uniform(self.dataset_neg_low, self.dataset_neg_high)
                            else:
                                Oneg = random.uniform(-self.dataset_neg_high, -self.dataset_neg_low)
#                            neg_Patch_1 = imgL[pos_X-self.inter_len:pos_X+self.inter_len+1, pos_Y-self.inter_len:pos_Y+self.inter_len+1]
                            pos_R_Y = int(pos_Y - d + Oneg)
                            if pos_R_Y-self.inter_len < 0 or pos_R_Y+self.inter_len+1 >= self.picLen:
                                continue
                            neg_Patch_2 = imgR[pos_X-self.inter_len:pos_X+self.inter_len+1, pos_R_Y-self.inter_len:pos_R_Y+self.inter_len+1]

                            patchNum += 1
                            if patchNum >= self.batchSize/2:
                                index1, index2 = self.nextIndex()

                                Batch_1[index1] = self.preprocess(pos_Patch_1)
                                Batch_2[index1] = self.preprocess(pos_Patch_2)
                                flag[index1] = 1.0

                                Batch_1[index2] = self.preprocess(pos_Patch_1)
                                Batch_2[index2] = self.preprocess(neg_Patch_2)
                                flag[index2] = 0.0

                                picName = str(batchNum) + '.npy'
                                np.save(self.Dir_1+picName, Batch_1)
                                np.save(self.Dir_2+picName, Batch_2)
                                np.save(self.flagDir+picName, flag)

                                batchNum += 1
                                patchNum = 0
                            elif patchNum == 1:
                                Batch_1 = np.zeros((self.batchSize, 9*9), dtype=np.float32)
                                Batch_2 = np.zeros((self.batchSize, 9*9), dtype=np.float32)
                                flag = np.zeros((self.batchSize, 1), dtype=np.float32)

                                self.indexNumInit()
                                index1, index2 = self.nextIndex()

                                Batch_1[index1] = self.preprocess(pos_Patch_1)
                                Batch_2[index1] = self.preprocess(pos_Patch_2)
                                flag[index1] = 1.0

                                Batch_1[index2] = self.preprocess(pos_Patch_1)
                                Batch_2[index2] = self.preprocess(neg_Patch_2)
                                flag[index2] = 0.0
                            else:
                                index1, index2 = self.nextIndex()

                                Batch_1[index1] = self.preprocess(pos_Patch_1)
                                Batch_2[index1] = self.preprocess(pos_Patch_2)
                                flag[index1] = 1.0

                                Batch_1[index2] = self.preprocess(pos_Patch_1)
                                Batch_2[index2] = self.preprocess(neg_Patch_2)
                                flag[index2] = 0.0

                            if batchNum >= 30:
                                break

    def next_batch(self):
        self.nextBatchNum += 1
        picName = str(self.nextBatchNum) + '.npy'
        Batch_1 = np.load(self.Dir_1+picName).astype(np.float32)
        Batch_2 = np.load(self.Dir_2+picName).astype(np.float32)
        flag = np.load(self.flagDir+picName).astype(np.float32)
        return Batch_1,Batch_2,flag

if __name__=="__main__":
    ## KITTI 2012 2015
    dataset_neg_low = 4
    dataset_neg_high = 10
    dataset_pos = 1
    input_patch_size = (9,9)
    inter_len = 4
    batchSize = 128

    preDir = 'E:/KITTI_DataSet/KITTI2012/data_stereo_flow/training/'
    resultDir = 'E:/StereoVision/Code/KITTIPatch/'

    myKITTI_Patch = KITTI_Patch(preDir, resultDir,
                                dataset_neg_low, dataset_neg_high,
                                dataset_pos, input_patch_size,
                                batchSize)
    myKITTI_Patch.create_batch()
    Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
#    myKITTI_Patch.indexNumInit()
#    a,b = myKITTI_Patch.nextIndex()
#    aa = myKITTI_Patch.indexNumList
#
#    c,d = myKITTI_Patch.nextIndex()
#    cc = myKITTI_Patch.indexNumList



