# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 19:17:20 2017

@author: MagicWang
"""

import os
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
import time
import base64
import threading
from stereoServer import StereoMatch


UPLOAD_FOLDER='./upload'
ALLOWED_EXTENSIONS = set(['txt','png','jpg','xls','JPG','PNG','xlsx','gif','GIF'])

stereoFlag = False
dispName = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024


def stereo():
    global stereoFlag
    while True:
        if stereoFlag:
            referPic = "./upload/tmp_L.png"
            targetPic = "./upload/tmp_R.png"
            global dispName
            dispPic = "./upload/" + dispName
            netDir = "./save/0316/10"
            print "Begin"
            match = StereoMatch(referPic, targetPic, netDir, dispPic)
            match.stereomatch()
            stereoFlag = False

# 用于判断文件后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return "ok"
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/api_begin/<filename>', methods=['GET', 'POST'])
def beginStereo(filename):
    if request.method == 'POST':
        global dispName
        dispName = filename
        global stereoFlag
        stereoFlag = True
        return "ok"


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
    stereoThread = threading.Thread(target=stereo)
    stereoThread.start()
    app.run(host='0.0.0.0', port=7000)
