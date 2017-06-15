#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:08:29 2017

@author: magic
"""
import tensorflow as tf
import FASTCONFIG
from FASTKITTI_Patch import KITTI_Patch
import time
import os
import numpy as np

class FastNet:
    def __str__(self):
	return 'StereoNet of Tensorflow'

    __repr__ = __str__
    
    def __init__(self, epochs, iteration,
                 num_conv_feature_maps, num_fc_units,
                 learning_rate, momentum,
                 weight_decay, saveDir):
        self.epochs = epochs
        self.iteration = iteration
        self.num_conv_feature_maps = num_conv_feature_maps
        self.num_fc_units = num_fc_units
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.saveDir = saveDir
        self.variables_dict = {
                  "conv1_weights": self.weight_with_decay(name="conv1_weights", shape=[3, 3,1, num_conv_feature_maps]),
                  "conv1_biases": tf.Variable(tf.constant(0.1, shape=[num_conv_feature_maps]), 
                          name="conv1_biases"),
                  "conv2_weights": self.weight_with_decay(name="conv2_weights", shape=[3, 3,num_conv_feature_maps, num_conv_feature_maps]),
                  "conv2_biases": tf.Variable(tf.constant(0.1, shape=[num_conv_feature_maps]), 
                          name="conv2_biases"),
                  "conv3_weights": self.weight_with_decay(name="conv3_weights", shape=[3, 3,num_conv_feature_maps, num_conv_feature_maps]),
                  "conv3_biases": tf.Variable(tf.constant(0.1, shape=[num_conv_feature_maps]), 
                          name="conv3_biases"),
                  "conv4_weights": self.weight_with_decay(name="conv4_weights", shape=[3, 3,num_conv_feature_maps, num_conv_feature_maps]),
                  "conv4_biases": tf.Variable(tf.constant(0.1, shape=[num_conv_feature_maps]), 
                          name="conv4_biases"),
                  }
        print "StereoNet Init"
      
    def weight_with_decay(self, name, shape):
        var = tf.Variable(tf.truncated_normal(shape,stddev=0.1), name=name)
        decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, name='weight_loss')
        tf.add_to_collection('losses', decay)
        return var
            
    def conv2d(self, x, W):
        ## 4D输入和4D过滤（卷积核），填充模式same影响输出图像大小
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def subNet(self, input_images):
        with tf.name_scope('Conv_Layer_1'):
            conv1 = tf.nn.relu(self.conv2d(input_images, self.variables_dict["conv1_weights"])+
                               self.variables_dict["conv1_biases"])
        with tf.name_scope('Conv_Layer_2'):
            conv2 = tf.nn.relu(self.conv2d(conv1, self.variables_dict["conv2_weights"])+
                               self.variables_dict["conv2_biases"])
        with tf.name_scope('Conv_Layer_3'):
            conv3 = tf.nn.relu(self.conv2d(conv2, self.variables_dict["conv3_weights"])+
                               self.variables_dict["conv3_biases"])
        with tf.name_scope('Conv_Layer_4'):
            conv4 = self.conv2d(conv3, self.variables_dict["conv4_weights"])+self.variables_dict["conv4_biases"]
        reshaped = tf.reshape(conv4, [-1, 9*9*self.num_conv_feature_maps])
        normal = tf.nn.l2_normalize(reshaped, 1)
        return normal
    
    def stereo_Train(self):
        preDir = FASTCONFIG.train_Data
        resultDir = FASTCONFIG.patch_Dir
        
        myKITTI_Patch = KITTI_Patch(preDir, resultDir, 
                                    FASTCONFIG.dataset_neg_low, FASTCONFIG.dataset_neg_high,
                                    FASTCONFIG.dataset_pos, FASTCONFIG.input_patch_size,
                                    FASTCONFIG.batchSize)
        ## time is too longer , do it before training
#        myKITTI_Patch.create_batch()
        with tf.name_scope('inputs'):
            batchLeft = tf.placeholder(tf.float32, [128, 81])
            batchRight = tf.placeholder(tf.float32, [128, 81])
            batch_flag = tf.placeholder(tf.float32, [128, 1])
            learning_rate = tf.placeholder(tf.float32, [])
        
        batch_image_1 = tf.reshape( batchLeft, [-1,9,9,1])
        batch_image_2 = tf.reshape( batchRight, [-1,9,9,1])
        
        with tf.name_scope('Left_SubNet'):
            subNet_1 = self.subNet(batch_image_1)
        with tf.name_scope('Right_SubNet'):
            subNet_2 = self.subNet(batch_image_2)
        
        with tf.name_scope('Dot_Product'):
            netout = tf.reduce_sum(tf.multiply(subNet_1, subNet_2), 1)
        
        with tf.name_scope('Loss'):
            batch = tf.gather(netout, tf.to_int32(batch_flag))
            correct_id = tf.range(0, 64)
            correct_id = tf.reshape(correct_id, [64,1])
            wrong_id = tf.range(64, 128)
            wrong_id = tf.reshape(wrong_id, [64,1])
            correctpatch = tf.gather_nd(batch, correct_id)
            wrongpatch = tf.gather_nd(batch, wrong_id)
            toLoss = tf.maximum(0.0, 0.2 + wrongpatch - correctpatch)
            hingeLoss = tf.reduce_sum(toLoss)
            tf.add_to_collection('losses', hingeLoss)
            Loss = tf.add_n(tf.get_collection('losses'), name='Loss')
        tf.summary.scalar('Loss', Loss)
        
        with tf.name_scope('Train'):
            train_step = tf.train.MomentumOptimizer(learning_rate, self.momentum).minimize(Loss)
        with tf.name_scope('Accuracy_Eval'): 
            net_prediction_1 = tf.round(correctpatch)
            net_prediction_2 = tf.round(wrongpatch)
            correct_prediction = tf.equal(net_prediction_1, 1.0)
            wrong_prediction = tf.equal(net_prediction_2, 0.0)
            accuracy_1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_2 = tf.reduce_mean(tf.cast(wrong_prediction, tf.float32))
            accuracy = (accuracy_1+accuracy_2)/2
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.scalar('LearningRate', learning_rate)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.saveDir + '/log/train', sess.graph)
        test_writer = tf.summary.FileWriter(self.saveDir + '/log/test')
        sess.run(tf.global_variables_initializer())
#        saver.restore(sess, "./save/0325/10/stereo.ckpt")

        print "======Train Begin======"
        t0 = time.clock()
        for e in range(self.epochs):
            print "=======HERE=========="
            print "time: " + str((time.clock()-t0)/60) + " minutes"
            t0 = time.clock()
            thisDir = "/"+str(e)
            if not os.path.exists(self.saveDir+thisDir):
                print "Create "+str(e)
                os.mkdir(self.saveDir+thisDir)
            save_path = saver.save(sess, self.saveDir + thisDir +"/stereo.ckpt")
            print("Model saved in file: %s" % save_path)
            myKITTI_Patch.nextBatchNum = 0
            myKITTI_Patch.Dir_1 = './FastKITTIPatch/Batch_1/'
            myKITTI_Patch.Dir_2 = './FastKITTIPatch/Batch_2/'
            myKITTI_Patch.flagDir = './FastKITTIPatch/Flag/'
            print "Reset KITTI 2012"
            print "=======HERE=========="
            if e == 0:
                self.learning_rate = self.learning_rate/10
            if e == 23:
                self.learning_rate = self.learning_rate/2
            for i in range(self.iteration):
                 Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
                 if i % 20000 == 0:
                     if i % 20000 == 0:
                         hinge, train_accuracy,train_Loss,summary = sess.run([hingeLoss, accuracy, Loss, merged],feed_dict = {
                                         batchLeft: Batch_1, batchRight: Batch_2, 
                                         batch_flag: flag, learning_rate: self.learning_rate})
                         train_writer.add_summary(summary, i)
                         print "epoch %d, step %d : \ntraining accuracy %s, hinge %s, Loss %s"%(e, i, train_accuracy, hinge, train_Loss)
                     else:
                         summary = sess.run(merged, feed_dict = {
                                           batchLeft: Batch_1, batchRight: Batch_2,
                                           batch_flag: flag, learning_rate: self.learning_rate})
                         train_writer.add_summary(summary, i)
                 sess.run(train_step, feed_dict = {
                                             batchLeft: Batch_1, batchRight: Batch_2,
                                             batch_flag: flag, learning_rate: self.learning_rate})
            newKITTI_Patch = KITTI_Patch(preDir, resultDir, 
                            FASTCONFIG.dataset_neg_low, FASTCONFIG.dataset_neg_high,
                            FASTCONFIG.dataset_pos, FASTCONFIG.input_patch_size,
                            FASTCONFIG.batchSize)
#            # little change every epochs
            newKITTI_Patch.create_batch()
            if e < (self.epochs-1):
                print "======Eval Begin======"
                accuracySum = 0.0
                hingeSum = 0.0
                min_accuacy = 1.0
                underFive = 0
                for i in range(4000):
                    Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
                    hinge, train_accuracy = sess.run([hingeLoss, accuracy], feed_dict = {
                                                                 batchLeft: Batch_1, batchRight: Batch_2,
                                                                 batch_flag: flag, learning_rate: self.learning_rate})
                    if min_accuacy > train_accuracy:
                        min_accuacy = train_accuracy
                    if train_accuracy < 0.5:
                        underFive += 1
                    accuracySum = accuracySum +  train_accuracy
                    hingeSum = hingeSum+hinge
                print "accuray Eval Result: "+ str(accuracySum/4000)
                print "minium accuracy: "+ str(min_accuacy)
                print "accuracy < 0.5 is " + str(underFive)
                print "hinge loss : " + str(hingeSum/4000)
                 
        save_path = saver.save(sess, self.saveDir + "/stereo.ckpt")
        print("Model saved in file: %s" % save_path)
        
        print "======Final Eval Begin======"
        accuracySum = 0.0
        min_accuacy = 1.0
        underFive = 0
        for i in range(4000):
#            myKITTI_Patch.nextBatchNum = 10000
            Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
            train_accuracy = sess.run(accuracy, feed_dict = {
                                                         batchLeft: Batch_1, batchRight: Batch_2,
                                                         batch_flag: flag, learning_rate: self.learning_rate})
            if min_accuacy > train_accuracy:
                min_accuacy = train_accuracy
            if train_accuracy < 0.5:
                underFive += 1
            if i % 10 == 0:
                summary = sess.run(merged, feed_dict = {
                                                       batchLeft: Batch_1, batchRight: Batch_2,
                                                       batch_flag: flag, learning_rate: self.learning_rate})
                test_writer.add_summary(summary, i)
            accuracySum = accuracySum +  train_accuracy
        
        print "accuray Eval Result: "+ str(accuracySum/4000)
        print "minium accuracy: "+ str(min_accuacy)
        print "accuracy < 0.5 is " + str(underFive)
    
    def preprocess(self, mat):
        reshapeMat = tf.reshape(mat, [-1, 81])
        mean, variance= tf.nn.moments(reshapeMat, [1])
        normal = tf.div(reshapeMat-tf.reshape(mean,[-1,1]), tf.reshape(tf.sqrt(variance),[-1,1]))
        output = tf.reshape(normal, [-1,9,9,1])
        return output
        
    def lineNet(self, d_range, picLen):
        picL = tf.placeholder(tf.float32, [9, picLen])
        picR = tf.placeholder(tf.float32, [9, picLen])
        Batch_Refer = []
        Batch_Target = []
        
        for le in range(picLen-8):
            Batch_Refer.append(tf.slice(picL, [0,le], [9,9]))
            Batch_Target.append(tf.slice(picR, [0,le], [9,9]))
        Line_Refer = tf.stack(Batch_Refer)
        Line_Target = tf.stack(Batch_Target)

        batch_image_1 = self.preprocess(Line_Refer)
        batch_image_2 = self.preprocess(Line_Target)
        
        with tf.name_scope('Left_SubNet'):
            subNet_1 = self.subNet(batch_image_1)
        with tf.name_scope('Right_SubNet'):
            subNet_2 = self.subNet(batch_image_2)
            
        subRev_1 = tf.reverse(subNet_1, [True, False])
        subRev_2 = tf.reverse(subNet_2, [True, False])
        
        with tf.name_scope('LeftDisp'):
            L_disp = []
            line_cost = []
            for i in range(picLen-8):
                if picLen-8 - i > d_range:
                    size = d_range
                else:
                    size = picLen -8 - i
                L_sub_2 = tf.slice(subRev_1, [i,0], [1, 9*9*self.num_conv_feature_maps])
                L_sub_1 = tf.slice(subRev_2, [i,0], [size, 9*9*self.num_conv_feature_maps])
                L_dot = tf.reduce_sum(tf.multiply(L_sub_1, L_sub_2), 1)
                L_dot_max = tf.arg_max(L_dot,0)
                line_pad = tf.pad(L_dot, [[0, d_range-size]])
                line_cost.append(line_pad)
                L_disp.append(L_dot_max)
            NetCost = tf.stack(line_cost)
            NetCost = tf.negative(NetCost)
            NetCost = tf.reverse(NetCost, [True, False])
            L_disp = tf.stack(L_disp)
            L_disp = tf.reverse(L_disp, [True])
        
        with tf.name_scope('RightDisp'):
            R_disp = []
            for i in range(picLen-8):
                if picLen-8 - i > d_range:
                    size = d_range
                else:
                    size = picLen -8 - i
                R_sub_2 = tf.slice(subNet_2, [i,0], [1, 9*9*self.num_conv_feature_maps])
                R_sub_1 = tf.slice(subNet_1, [i,0], [size, 9*9*self.num_conv_feature_maps])
                R_dot = tf.reduce_sum(tf.multiply(R_sub_1, R_sub_2), 1)
                #做不做，看效果
#                R_dot = tf.select(R_dot < 0.5, tf.zeros_like(R_dot),R_dot)
                R_dot_max = tf.arg_max(R_dot,0)
                R_disp.append(R_dot_max)
            R_disp = tf.stack(R_disp)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.saveDir + "/stereo.ckpt")
        return sess, NetCost, L_disp, R_disp,picL, picR
        
    def fullNet(self, d_range, picWidth, picLen):
#        picL = tf.placeholder(tf.float32, [picWidth, picLen])
#        picR = tf.placeholder(tf.float32, [picWidth, picLen])
        Batch_Refer = []
        Batch_Target = []
        allDisp = []
#        filename = [str(i)+'.tfrecords' for i in range(100) ]
        
        reader = tf.TFRecordReader()
        num = 0        
        for w in range(picWidth-8):
            num += 1
            name = str(num) + '.tfrecords'
            filename_queue = tf.train.string_input_producer(['./save/'+name])
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,features = {
                        'image_left':tf.FixedLenFeature([], tf.string),
                        'image_right':tf.FixedLenFeature([], tf.string)})
            picL = tf.decode_raw(features['image_left'], tf.uint8)
            picR = tf.decode_raw(features['image_right'], tf.uint8)
            picL = tf.reshape(picL, [9, picLen])
            picL = tf.to_float(picL)
            picR = tf.reshape(picR, [9, picLen])
            picR = tf.to_float(picR)
            for le in range(picLen-8):
                Batch_Refer.append(tf.slice(picL, [0,le], [9,9]))
                Batch_Target.append(tf.slice(picR, [0,le], [9,9]))
            Line_Refer = tf.pack(Batch_Refer)
            Line_Target = tf.pack(Batch_Target)
        
            batch_image_1 = self.preprocess(Line_Refer)
            batch_image_2 = self.preprocess(Line_Target)
            
            with tf.name_scope('Left_SubNet'):
                subNet_1 = self.subNet(batch_image_1)
            with tf.name_scope('Right_SubNet'):
                subNet_2 = self.subNet(batch_image_2)
                
            subRev_1 = tf.reverse(subNet_1, [True, False])
            subRev_2 = tf.reverse(subNet_2, [True, False])
            
            disp = []
#            subRev_1 = ReferAll[w]
#            subRev_2 = TargetAll[w]
            for i in range(picLen-8):
                if picLen-8 - i > d_range:
                    size = d_range
                else:
                    size = picLen -8 - i
                sub_1 = tf.slice(subRev_2, [i,0], [size, 9*9*self.num_conv_feature_maps])
                sub_2 = tf.slice(subRev_1, [i,0], [1, 9*9*self.num_conv_feature_maps])
    #            sub_2 = tf.tile(sub_one, [size, 1])
                dot = tf.reduce_sum(tf.multiply(sub_1, sub_2), 1)
                #做不做，看效果
    #            dot = tf.select(dot < 0.5, tf.zeros_like(dot), dot)
                dot_max = tf.arg_max(dot,0)
                if i == 0:
                    shape =  tf.shape(dot)
    #            maxVal = tf.reduce_max(dot)
    #            realD =tf.where(tf.less(maxVal, 0.5), 0, dot_max)
    #            if tf.less(maxVal, 0.5):
    #                dot_max = 0
                disp.append(dot_max)
            disp = tf.pack(disp)
            disp = tf.reverse(disp, [True])
            allDisp.append(disp)
        allDisp = tf.pack(allDisp)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.saveDir + "/stereo.ckpt")
        return sess, shape, allDisp 
        
        
            
if __name__=="__main__":
    stereo = FastNet(FASTCONFIG.epochs, FASTCONFIG.iteration,
                     FASTCONFIG.num_conv_feature_maps, FASTCONFIG.num_fc_units,
                     FASTCONFIG.learning_rate, FASTCONFIG.momentum,
                     FASTCONFIG.weight_decay, FASTCONFIG.save_Dir)
    stereo.stereo_Train()
#    stereo.netbuild()
#    stereo.stereo_Eval()
    


