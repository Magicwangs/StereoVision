## A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation
一个训练视差，光流，场景流估计卷积网络的大型数据集

双目匹配与视差计算 立体匹配主要是通过找出每对图像间的对应关系，根据三角测量原理，得到视差图；在获得了视差信息后，根据投影模型很容易地可以得到原始图像的深度信息和三维信息。立体匹配技术被普遍认为是立体视觉中最困难也是最关键的问题，主要是以下因素的影响： （1） 光学失真和噪声（亮度、色调、饱和度等失衡） （2） 平滑表面的镜面反射 （3） 投影缩减（Foreshortening） （4） 透视失真（Perspective distortions） （5） 低纹理（Low texture） （6） 重复纹理（Repetitive/ambiguous patterns） （7） 透明物体 （8） 重叠和非连续 目前立体匹配算法是计算机视觉中的一个难点和热点，算法很多，但是一般的步骤是： A、匹配代价计算 匹配代价计算是整个立体匹配算法的基础，实际是对不同视差下进行灰度相似性测量。常见的方法有灰度差的平方SD（squared intensity differences），灰度差的绝对值AD（absolute intensity differences）等。另外，在求原始匹配代价时可以设定一个上限值，来减弱叠加过程中的误匹配的影响。以AD法求匹配代价为例，可用下式进行计算，其中T为设定的阈值。 图18 B、 匹配代价叠加 一般来说，全局算法基于原始...

## Referance
- [Blender](https://www.blender.org/)  
三维动画制作软件  

相机完全标定  
光流是指图像灰度模式的表面运动，是三维运动场在二维图像平面上的投
影，  
场景流是空间中场景运动形成的三维运动场。  
- [场景流](http://www.doc88.com/p-2344567483739.html)  
- [变分方法](http://baike.baidu.com/view/11584829.htm)  
RGBD：RGB+D(Depth)  
- [图像退化](http://zuoye.baidu.com/question/9ed9842fd9111c2544ebc95e485e2609.html)  
- []()  
- []()  
- []()  
