# stereoVision

## 超参数：
input_patch_size: 输入图像块大小，9x9
num_conv_layers：子网中卷积层的数量，4
conv_kernel_size：卷积核大小，3
num_conv_feature_maps：每层的特征映射数量，64

每个图像都通过减去平均值并除以像素强度值的标准差来预处理。立体图像对的左和右图分别预处理。

hinge损失最小化

batch大小：128
在训练前打乱

mini-batch gradient来减少损失  momentum为0.9
stochastic gradient descent就是MomentumOptimizer
精确架构采用交叉熵损失cross-entropy

14次迭代，快速架构初始学习率0.002，精确架构0.003
学习率在11次迭代时减少10倍

cross-validation优化

每个图像都通过减去平均值并除以像素强度值的标准差来预处理
Nan替换成0

siamese网络

快速架构采用余弦相似距离，计算相似性得分

## 精确架构
卷积+ReLu，串联，全连接+ReLu，全连接+sigmoid，相似性得分

二元交叉熵损失cross-entropy，看论文定义
mini-batch gradient来减少损失  momentum为0.9
stochastic gradient descent就是MomentumOptimizer

num_conv_layers：每个子网络中卷积层的数量，4
num_conv_feature_maps：每个层中的特征映射的数量，112
conv_kernel_size：卷积内核的大小，3
input_patch_size：图像块大小，9*9
num_fc_units：每个全连接层中的单元数，384
num_fc_layers：全连接层数，4


训练迭代14次，对于精确架构学习率的初始设置为0.003，
学习率在11次迭代时减少10倍

网络的初值很重要


## 结果可视觉化
可以用散点图来画
从log中提取出来数据。plt画
参考https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/3-3-visualize-result/

## tensorboard
后期可以考虑通过他来展示和理解网络
