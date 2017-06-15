# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:21:43 2017

@author: MagicWang
"""
from PyQt4 import QtCore, QtGui, uic
import sys
import cv2
import numpy as np
import threading
import requests
import Queue
import os
import Params
import time

running = False
dispPic = ""
capture_thread_1 = None
capture_thread_2 = None
form_class = uic.loadUiType("app.ui")[0]
L_q = Queue.Queue()
R_q = Queue.Queue()


def grab(cam, queue, width, height, fps):
    global running
#    while(1):
#        if running:
#            break
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while(running):
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        img = cv2.flip(img, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        frame["img"] = img

        if queue.qsize() < 2:
            queue.put(frame)
        else:
#            print queue.qsize()
            pass
    capture.release()

class OwnImageWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()


class MyWindowClass(QtGui.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.StartBtn.clicked.connect(self.start_clicked)
        self.CapBtn.clicked.connect(self.capture)
        self.UpBtn.clicked.connect(self.upload)
        self.DispBtn.clicked.connect(self.disparity)

        self.window_width = self.Left_ImgWidget.frameSize().width()
        self.window_height = self.Left_ImgWidget.frameSize().height()

        self.Left_ImgWidget = OwnImageWidget(self.Left_ImgWidget)
        self.Right_ImgWidget = OwnImageWidget(self.Right_ImgWidget)

        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")

        capture_thread_1.start()
        capture_thread_2.start()
        global running
        running = True

    # 添加点击事件，打印当前点的距离
    def callbackFunc(self, e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            text = "Depth: " + str(Params.threeD[y][x][2])
            self.textBrowser.append(text)

    def stereoRec(self, leftPic, rightPic):
        imgL_rectified = cv2.remap(leftPic, Params.left_map1, Params.left_map2, cv2.INTER_LINEAR)
        imgR_rectified = cv2.remap(rightPic, Params.right_map1, Params.right_map2, cv2.INTER_LINEAR)
        return imgL_rectified, imgR_rectified

    def project3D(self, disp):
        cv2.namedWindow("Disparity")
        cv2.setMouseCallback("Disparity", self.callbackFunc, None)
        disparity = disp.astype(np.float32)/256.0
        Params.threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32), Params.Q)
        disp = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        showDisp = cv2.applyColorMap(disp, cv2.COLORMAP_HSV)
        cv2.imshow("Disparity", showDisp)
        cv2.waitKey(0)

    def start_clicked(self):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.textBrowser.append("Start Camera")

        self.StartBtn.setEnabled(False)
        self.StartBtn.setText('Starting...')

    def capture(self):
        self.textBrowser.append("Capture Image")
        self.StartBtn.setEnabled(True)
        self.StartBtn.setText('Start')
        self.timer.stop()

        L_frame = L_q.get()
        L_Img = L_frame["img"]
        R_frame = R_q.get()
        R_Img = R_frame["img"]
        cv2.imwrite("./tmp/original_L.png", L_Img)
        cv2.imwrite("./tmp/original_R.png", R_Img)
        rec_L, rec_R = self.stereoRec(L_Img, R_Img)
        saveL = cv2.cvtColor(rec_L, cv2.COLOR_BGR2GRAY)
        saveR = cv2.cvtColor(rec_R, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("./tmp/tmp_L.png", saveL)
        cv2.imwrite("./tmp/tmp_R.png", saveR)

        img = rec_L
        img_height, img_width, img_colors = img.shape
        scale_w = float(self.window_width) / float(img_width)
        scale_h = float(self.window_height) / float(img_height)
        scale = min([scale_w, scale_h])
        if scale == 0:
            scale = 1
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        height, width, bpc = img.shape
        bpl = bpc * width
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.Left_ImgWidget.setImage(image)


        img = rec_R
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.Right_ImgWidget.setImage(image)


    def upload(self):
        self.textBrowser.append("Upload Image ...")
        url = "http://magicwang.ngrok.cc/"
        L_image = {'file': open('./tmp/tmp_L.png', 'rb')}
        R_image = {'file': open('./tmp/tmp_R.png', 'rb')}
        p1 = requests.post(url, files=L_image)
        p2 = requests.post(url, files=R_image)
        thisTime = time.strftime('%H%I%M%S',time.localtime(time.time()))
        global dispPic
        dispPic = str(thisTime)+".png"
        beginUrl = "http://magicwang.ngrok.cc/api_begin/" + dispPic
        p3 = requests.post(beginUrl)
        if p1.ok and p2.ok and p3.ok:
            self.textBrowser.append("Upload Success")
        else:
            self.textBrowser.append("Upload Failed")

    def disparity(self):
       # img = cv2.imread("./tmp/disp.png", flags=cv2.IMREAD_ANYDEPTH)
       # self.project3D(img)
        self.textBrowser.append("Getting Disparity...")
        global dispPic
        dispUrl = "http://magicwang.ngrok.cc/uploads/"+dispPic
        r = requests.get(dispUrl, stream=True)
        if r.ok:
            self.textBrowser.append("Successed Get")
            with open("./tmp/disp.png", 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                f.close()
            img = cv2.imread("./tmp/disp.png", flags=cv2.IMREAD_ANYDEPTH)
            self.project3D(img)
        else:
            self.textBrowser.append("Get Failed")

    def update_frame(self):
        if not L_q.empty():
            self.StartBtn.setText('Camera Alive')
            L_frame = L_q.get()
            img = L_frame["img"]
            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])
            if scale == 0:
                scale = 1
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            height, width, bpc = img.shape
            bpl = bpc * width
            L_image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)

            R_frame = R_q.get()
            img = R_frame["img"]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
            R_image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)

            self.Left_ImgWidget.setImage(L_image)
            self.Right_ImgWidget.setImage(R_image)

    def closeEvent(self, event):
        global running
        running = False


if __name__=="__main__":
    capture_thread_1 = threading.Thread(target=grab, args = (0, L_q, 640, 480, 30))
    capture_thread_2 = threading.Thread(target=grab, args = (1, R_q, 640, 480, 30))

    app = QtGui.QApplication(sys.argv)
    w = MyWindowClass(None)
    w.setWindowTitle('Magic Stereo Camera')
    w.setWindowIcon(QtGui.QIcon('./icon\icon.png'))
    w.show()
    app.exec_()