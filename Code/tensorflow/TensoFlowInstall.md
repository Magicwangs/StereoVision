# Linux
[Linux命令大全](http://man.linuxde.net/)
## linux ssh自动断线问题
https://www.coder4.com/archives/3751

## 连接服务器
ssh -X -l xhq -p 53866 ittun.com

## 当前目录下的文件个数
ls -l |grep "^-"|wc -l
## 文件夹大小
du -h .

## GPU使用情况
nvidia-smi

## yum因被锁定导致无法执行相关操作的解决方法
执行yum命令时，提示
```
Another app is currently holding the yum lock; waiting for it to exit...
```
等错误字样。这是因为yum被锁定无法使用导致的。
解决方案
```
rm -rf /var/run/yum.pid
```
执行以上命令，删除yum.pid强行解除锁定即可。

## 前后台切换
主要可以参考这两篇
[Linux 进程前后台切换|管理](https://segmentfault.com/a/1190000000349722)    
[Linux 进程后台运行的几种方式（screen）](https://segmentfault.com/a/1190000002607962)  
```
# 后台运行，输出记录在当前目录下的nohup.out中可以cat查看
nohup 命令 &

# 正在运行的命令切换后台
ctrl+z停止后
bg %1
jobs -l #查看进程
fg %1
** 1 只是标号，具体看是几号进程**
**建议不要再下载时使用这个，可能会下载不完全**

# 结束
kill %1

## 退出终端后再次进入，查看python进程
ps -ef|grep python

screen命令
screen -S yourname：开启新的窗口
screen -ls：列出当前所有的session
ctrl + A + D：退出
screen -r id：根据id恢复
screen -wipe：清理死掉的session
kill id：关闭对话
```

## proxychains + shadowsocks代理命令行
pip install shadowsocks
vi /etc/shadowsocks.json
写入
https://shadowsocks.com/download.html  
sslocal /etc/shadowsocks.json运行  

安装proxychains网上教程较多，难度也不高
http://shawnelee88.github.io/2015/07/10/proxychains-shadowsocks%E7%A5%9E%E5%99%A8/  

命令行代理
proxychains4 命令行

在浏览器中使用一般需要装代理插件，可自行google  

# pip安装
[pip常用命令](http://me.iblogc.com/2015/01/01/pip%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4/)
## pip国内源
可以参考清华大学的帮助  
https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
清华大学  
https://pypi.tuna.tsinghua.edu.cn/simple
豆瓣  
http://pypi.doubam.com/simple/
中国科学技术大学  
https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/

## yum
可以通过强制关掉yum进程：
rm -f /var/run/yum.pid

yum安装时，依赖出错，可以单独下载出错的依赖，可能是版本问题

# conda安装
## 寻找合适的conda包
anaconda search -t conda 包名
anaconda show jjhelmus/tensorflow（某个源）
上面show了之后会告诉你怎么装的

## 添加国内conda源
conda config --add channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/'
显示下载通道
conda config --set show_channel_urls yes
添加了国内清华大学的源，其他的你可以再找找
中科大的源http://mirrors.ustc.edu.cn/

## conda下载后本地安装
http://conda-test.pydata.org/docs/commands/install.html
根据conda的命令，似乎可以本地安装，还在尝试中
conda install --use-local 本地包
好吧，尝试失败了。。。。。。。。

## conda生成带有anaconda的环境
conda create -n name python=2.7 anaconda

# TensorFlowInstall
安装主要看的是官方的教程.
pip安装时清华大学的源上有最新的0.12的版本，也有tensorflow-gpu的版本。可以自己先search看看。
cuda的安装方法也基本按教程的走，建议cuda的toolkit采用网络安装，自己下载速度比较慢。
[官方英文教程-齐全](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#optional-install-cuda-gpus-on-linux)
[中文的翻译-落后](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html0)
[极客学院的翻译](http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/os_setup.html#install_cuda)
[github上的翻译](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/get_started/os_setup.md)

## 运行第一个范例——mnist手写字识别
运行后就自己一边等着吧，数据会下载在当前目录下的data中，下载数据很慢，建议放在后台进行  
## 关于变量的设置
export只是暂时的设置环境变量，需要永久设置变量
[设置环境变量永久生效 ](http://linuxso.com/linuxxitongguanli/1812.html)  


# opencv install
安装历时n天，中间断断续续，主要是安装的版本问题，最简单的2.4.10版本，可以直接通过conda安装。但存在很多问题。
window上一直用的是2.4.11的版本，感觉稍有问题，最终在linux上安装成功的是opencv3.1的版本，总体来说，应该没什么问题

windows上的opencv安装简单粗暴，直接把下载exe解压后，找到cv2.pyd，然后复制到python对应的site-packages下，就可以使用了。
[github上某windows教程](https://github.com/twtrubiks/FaceDetect/tree/master/How%20Install%20OpenCV%20in%20on%20Windows%20for%20Python)

linux上，需要先下载linux的源码包，可以从git下载某个版本，或是直接从官网给的链接下，不翻墙速度也还不错  
[官网教程](http://docs.opencv.org/3.1.0/dd/dd5/tutorial_py_setup_in_fedora.html)  
[Opencv3.1+python2.7的CentOS7安装](http://blog.csdn.net/daunxx/article/details/50506625)

cmake是需要注意，如果同时安装了gtk2和gtk3的话，需要设置参数，
参数可以通过cmake-gui查看
实践出真知，每台机子情况不同，cmake的问题主要是参数设置问题。







#
