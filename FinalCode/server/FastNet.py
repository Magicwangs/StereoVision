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
#                  "conv5_weights": self.weight_with_decay(name="conv5_weights", shape=[3, 3, num_conv_feature_maps, num_conv_feature_maps]),
#                  "conv5_biases": tf.Variable(tf.constant(0.1, shape=[num_conv_feature_maps]), 
#                          name="conv5_biases")
                  }
        print "StereoNet Init"
      
    def weight_with_decay(self, name, shape):
        var = tf.Variable(tf.truncated_normal(shape,stddev=0.1), name=name)
        decay = tf.mul(tf.nn.l2_loss(var), self.weight_decay, name='weight_loss')
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
#        with tf.name_scope('Conv_Layer_5'):
#            conv5 = self.conv2d(conv4, self.variables_dict["conv5_weights"])+self.variables_dict["conv5_biases"]
        reshaped = tf.reshape(conv4, [-1, 9*9*self.num_conv_feature_maps])
        normal = tf.nn.l2_normalize(reshaped, 1)
#        Sqrt = tf.reduce_sum(tf.sqrt(reshaped), 1,  keep_dims=True)
#        normal = tf.div(reshaped, Sqrt)
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
            netout = tf.reduce_sum(tf.mul(subNet_1, subNet_2), 1)
        
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
        tf.scalar_summary('Loss', Loss)
        
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
        tf.scalar_summary('Accuracy', accuracy)
        tf.scalar_summary('LearningRate', self.learning_rate)
        
        sess = tf.Session()
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(self.saveDir + '/log/train', sess.graph)
        test_writer = tf.train.SummaryWriter(self.saveDir + '/log/test')
        sess.run(tf.global_variables_initializer())
#        saver.restore(sess, "./save/0204/stereo.ckpt")

        print "======Train Begin======"
        t0 = time.clock()
        for e in range(self.epochs):
            print "time: " + str((time.clock()-t0)/60)
            t0 = time.clock()
            save_path = saver.save(sess, self.saveDir + "/stereo.ckpt")
            print("Model saved in file: %s \n" % save_path)
            myKITTI_Patch.nextBatchNum = 0
            myKITTI_Patch.Dir_1 = './FastKITTIPatch/Batch_1/'
            myKITTI_Patch.Dir_2 = './FastKITTIPatch/Batch_2/'
            myKITTI_Patch.flagDir = './FastKITTIPatch/Flag/'
            print "Reset KITTI 2012"

            if e == 10:
                self.learning_rate = self.learning_rate/10
            if e == 15:
                self.learning_rate = self.learning_rate/5
            for i in range(self.iteration):
                 Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
                 if i % 10000 == 0:
                     hinge, train_accuracy,train_Loss,summary = sess.run([hingeLoss, accuracy, Loss, merged],feed_dict = {
                                                                 batchLeft: Batch_1, batchRight: Batch_2, 
                                                                 batch_flag: flag, learning_rate: self.learning_rate})
                     train_writer.add_summary(summary, i)
                     print "epoch %d, step %d : \ntraining accuracy %s, hinge %s, Loss %s"%(e, i, train_accuracy, hinge, train_Loss)
                     
                 sess.run(train_step, feed_dict = {
                                             batchLeft: Batch_1, batchRight: Batch_2,
                                             batch_flag: flag, learning_rate: self.learning_rate})
            newKITTI_Patch = KITTI_Patch(preDir, resultDir, 
                            FASTCONFIG.dataset_neg_low, FASTCONFIG.dataset_neg_high,
                            FASTCONFIG.dataset_pos, FASTCONFIG.input_patch_size,
                            FASTCONFIG.batchSize)
            # little change every epochs
            newKITTI_Patch.create_batch()
                 
        save_path = saver.save(sess, self.saveDir + "/stereo.ckpt")
        print("Model saved in file: %s" % save_path)
        
        print "======Eval Begin======"
        accuracySum = 0.0
        min_accuacy = 1.0
        for i in range(4000):
#            myKITTI_Patch.nextBatchNum = 10000
            Batch_1,Batch_2,flag = myKITTI_Patch.next_batch()
            train_accuracy = sess.run(accuracy, feed_dict = {
                                                         batchLeft: Batch_1, batchRight: Batch_2,
                                                         batch_flag: flag, learning_rate: self.learning_rate})
            if min_accuacy > train_accuracy:
                min_accuacy = train_accuracy
            if i % 10 == 0:
                summary = sess.run(merged, feed_dict = {
                                                       batchLeft: Batch_1, batchRight: Batch_2,
                                                       batch_flag: flag, learning_rate: self.learning_rate})
                test_writer.add_summary(summary, i)
            accuracySum = accuracySum +  train_accuracy
        
        print "accuray Eval Result: "+ str(accuracySum/4000)
        print "minium accuracy: "+ str(min_accuacy)
    
    def preprocess(self, mat):
        reshapeMat = tf.reshape(mat, [-1, 81])
        mean, variance= tf.nn.moments(reshapeMat, [1])
        normal = tf.div(reshapeMat-tf.reshape(mean,[-1,1]), tf.reshape(tf.sqrt(variance),[-1,1]))
        output = tf.reshape(normal, [-1,9,9,1])
        return output
    
    def netbuild(self):
        with tf.name_scope('inputs'):
            batchLeft = tf.placeholder(tf.float32, [None, 9, 9])
            batchRight = tf.placeholder(tf.float32, [None, 9, 9])
            subNet_1 = tf.placeholder(tf.float32, [None, 9*9*self.num_conv_feature_maps])
            subNet_2 = tf.placeholder(tf.float32, [None, 9*9*self.num_conv_feature_maps])
        
        batch_image_1 = self.preprocess(batchLeft)
        batch_image_2 = self.preprocess(batchRight)
        
        with tf.name_scope('Left_SubNet'):
            subNet_1 = self.subNet(batch_image_1)
        with tf.name_scope('Right_SubNet'):
            subNet_2 = self.subNet(batch_image_2)
        
        with tf.name_scope('Dot_Product'):
            netout = tf.reduce_sum(tf.mul(subNet_1, subNet_2), 1)
        
#        disp = tf.arg_max(netout,0)
#        maxVal = tf.reduce_max(netout)
            
        sess = tf.Session()
        saver = tf.train.Saver()
        merged = tf.merge_all_summaries()
        sess.run(tf.global_variables_initializer())
#        saver.restore(sess, "./save/0204/stereo.ckpt")
        saver.restore(sess, self.saveDir + "/stereo.ckpt")

        return sess, netout, subNet_1, subNet_2, batchLeft, batchRight
            
if __name__=="__main__":
    stereo = FastNet(FASTCONFIG.epochs, FASTCONFIG.iteration,
                     FASTCONFIG.num_conv_feature_maps, FASTCONFIG.num_fc_units,
                     FASTCONFIG.learning_rate, FASTCONFIG.momentum,
                     FASTCONFIG.weight_decay, FASTCONFIG.save_Dir)
    stereo.stereo_Train()
#    stereo.netbuild()
#    stereo.stereo_Eval()
    


