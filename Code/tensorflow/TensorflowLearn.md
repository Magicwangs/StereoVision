## TensorflowLearning
Learning from [莫烦Python](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/)  

## 神经网络
机器学习 其实就是让电脑不断的尝试模拟已知的数据. 他能知道自己拟合的数据离真实的数据差距有多远, 然后不断地改进自己拟合的参数,提高拟合的相似度.

## Tensorflow
先定义神经网络的结构，然后把数据放入这个网络中去运算和训练
核心是结构
损失计算，优化器（梯度下降），反复训练，使权重和偏置趋向于正确的值

激活函数：对某些特征进行强化或者减弱。筛选一下。relu和sigmoid都是

优化器optimizer：

`epoch`：1个epoch等于使用训练集中的全部样本训练一次；
`iteration`：1个iteration等于使用batchsize个样本训练一次；

## CNN
[google视频](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/63796332430923)  
平移不变性，权重共享

## checkpoint
`查看checkpoint的内容`
python /home/magic/anaconda2/envs/tensorflow/lib/python2.7/site-packages/tensorflow/python/tools/inspect_checkpoint.py --file_name=./save/stereo.ckpt







###
