#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:46:09 2017

@author: magic
"""
import FASTCONFIG
import os
from NewFastStereoMatch import StereoMatch
import argparse

if __name__ == "__main__":
    # 在cmd中输入python standard.py --engine Bing
    parser=argparse.ArgumentParser()
    parser.add_argument(        
        '-e','--engine',  #变量名称由--后面的字符决定
        default='1',
        help="number"
        )
    args=parser.parse_args()
#    print args.engine
    
    picDir = FASTCONFIG.train_Data
    referDir = picDir + "/image_0/"
    targetDir = picDir + "/image_1/"
    
    allNetDir = "./save/0325/"
    picList = ['000000_10.png','000001_10.png','000002_10.png','000191_10.png','000192_10.png','000193_10.png']    
    i = args.engine
    final_dispDir = './disp/'+str(i)+"/"
    left_dispDir = final_dispDir+"/leftdisp/"
    right_dispDir = final_dispDir + "/rightdisp/"
    if not os.path.exists(final_dispDir):    
        os.mkdir(final_dispDir)
        os.mkdir(left_dispDir)
        os.mkdir(right_dispDir)
    netDir = allNetDir+str(i)
    match = StereoMatch(referDir, targetDir, final_dispDir, left_dispDir, right_dispDir, netDir, picList)
    out = match.stereomatch()
#    final_dispDir = './disp/'
#    left_dispDir = "./disp/leftdisp/"
#    right_dispDir = None
#    right_dispDir = './disp/rightdisp/'
#    netDir = allNetDir
#    match = StereoMatch(referDir, targetDir, final_dispDir, left_dispDir, right_dispDir, netDir)
#    out = match.stereomatch()
