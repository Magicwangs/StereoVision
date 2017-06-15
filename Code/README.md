## KITTI数据集
###关于KITTI的文件夹
`标号`:6位数字的序号_两位数字的帧号
`image_0`:灰度图,左图,left
`image_1`:灰度图,右图,right
`colored`:有RGB颜色的图
`disp_noc`:视差图，non-occluded，不考虑遮挡,范围0-256
`disp_occ`:视差图，occluded，考虑遮挡区域
`flow_noc`:光流场，non-occluded，不考虑遮挡
`flow_occ`:光流场，occluded，考虑遮挡区域

2012产生数据：304000-3000（nan太多超过1/2） 实际使用290000 
2015产生数据：277000  实际使用26000
### 图片格式
视差图必须保存为uint16（unsigned short）
uint8：unsigned char 0-256
uint16：unsigned short 0-65536
## 视差图光流场读取
视差误差计算，参考开发套件中的matlab代码
img2 = cv2.imread('00.png', flags=cv2.CV_16U)  
img3 = cv2.imread('no_occ.png', flags=cv2.IMREAD_ANYDEPTH)  
img2和img3两种方法读取都是读取单声道的uint16.读取视差图  
img4 = cv2.imread('flow.png', flags=cv2.IMREAD_UNCHANGED)  
img4是读取多声道的uint6,用于读取光流场  
[Open a multichannel image in Python OpenCV2](http://stackoverflow.com/questions/17534489/open-a-multichannel-image-in-python-opencv2)  

img3 = img2.astype('float64')/256  
视差图提取后必须转换为浮点型计算误差  
## 数据集矫正
立体数据集中的数据矫正可参考论文，与opencv的矫正办法相似
4个参数表示不同的相机的重投影矩阵，
P0：left grayscale
P1：right grayscale
P2：left color
P3：right color
只有投影矩阵？三维点到立体矩阵

## 双目标定
从相机中直接提取的图像，由于Distortion(Optics)，需要进行内外参标定,常用棋盘标定法
[问题参考](http://stackoverflow.com/questions/40977325/python-opencv-camera-calibration-cvimshow-error)

双目立体标定和单目立体标定不同！！！！
双目标定：cv2.stereoCalibrate
不再使用opencv立体标定，采用Matlab的Camera Calibration Toolbox
[使用OpenCV/python进行双目测距](http://www.cnblogs.com/zhiyishou/p/5767592.html)
[摄像机标定研究(Matlab+opencv+emgucv)](http://www.voidcn.com/blog/t247555529/article/p-6210675.html)

getChessPic：VideoCapture()
径向畸变和横向畸变设置，一个三个参数？（可以是两个参数），一个开启
双目标定后，要观察reprojection error（对象点和图像点之间的距离，越小越好）尽量聚敛，去掉过高的图再矫正
保持图片在10幅左右
获取参数：
导出参数，在matlab中查看参数  
摄像机矩阵的转制：stereoParams.CameraParameters1.IntrinsicMatrix
畸变系数（不宜超过四个，一般把第五个设为0）
两个畸变参数组合
[关于各个参数含义](http://cn.mathworks.com/help/vision/ref/cameraparameters-class.html)


## 双目矫正
Bouguet算法，参考《学习opencv》P497
在configs中的矫正参数解释：《学习opencv》P495
R，T:将右摄像机图像平面旋转到左摄像机图像平面的旋转矩阵和平移矩阵
R1：左摄像机（对右摄像机）上由摄像机到3D点的旋转矩阵
P1：左摄像机相应的坐标系下3D点P的位置


## 双目匹配
匹配在同一行，左图像某点在右图的匹配一定出现在同一行，相同位置的左边  

`可以列一张表格，显示blocksize和numdisparity的区别`

《学习opencv》P507 三个步骤
opencv自带SAD窗的视差匹配情况：

自带的SGBM方法：

matlab的视差计算情况
30%的错误率 matlab_Cailbration.m
默认是SGBM。
调节blocksize到5，可以降到16%左右

先做到深度图，3D图的问题，后续再考虑
matlab，3D重构
[双目测距与三维重建的OpenCV实现问题集锦（四）](http://blog.csdn.net/chenyusiyuan/article/details/5970799)
