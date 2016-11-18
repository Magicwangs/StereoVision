## Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches
通过训练比较图像补丁的卷积神经网络来立体视觉匹配

立体匹配算法主要是通过建立一个能量代价函数，通过此能量代价函数最小化来估计像素点视差值。立体匹配算法的实质就是一个最优化求解问题，通过建立合理的能量函数，增加一些约束，采用最优化理论的方法进行方程求解，这也是所有的病态问题求解方法。


立体匹配(Stereo Matching)算法主要分为全局(global)和局部(local)算法，其中全局算法步骤是：匹配代价计算(matching cost computation)，视差计算(disparity computation)和视差精化(disparity refinement)；局部算法步骤是：匹配代价计算(matching cost computation)，代价聚合(cost aggregation)，视差计算(disparity computation)和视差精化(disparity refinement)。所有的立体匹配算法只能计算到前2/3步，也就是视差精化之前，得到了初始视差图(initial disparity map)，你会发现它有很多的黑色区域，特别是左视差图的最左边，因为那是遮挡区域(occluded regions)，而立体匹配的算法只能计算出非遮挡区域(non-occluded regions)的视差，所以要将右视差图计算出来并进行左右一致性检查(Left-Regiht Crosscheck)来进行遮挡区域的填充，LeCun的论文虽然是采用了机器学习的方法产生视差图，但是仍然避免不了遮挡区域的处理问题，所以会引入亚像素增强(subpixel enhancement)这个策略(strategy)来进行视差精化(也可以叫后处理-post processing)。

## Referance
- [数据集扩增，泛化误差](http://chuansong.me/n/1834612)  
- [统计变换](http://blog.csdn.net/qianchenglenger/article/details/19931259)  
- [迁移学习](http://blog.csdn.net/miscclp/article/details/6339456)  
- [泛化性能](https://www.jianyujianyu.com/machine-learning-performance-evaluation/)  
泛化类似于推广的意思，对于从未在生成或训练网络时使用过的测试数据，若网络计算的输入-输出映射对它们来说都是正确的，或接近与正确，就认为网络的泛化是很好的。  
- [Github源码lua语言](https://github.com/jzbontar/mc-cnn)  
