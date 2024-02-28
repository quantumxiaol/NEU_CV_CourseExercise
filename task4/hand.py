import sys

from hand1 import Ui_Form

import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import cv2
import mediapipe as mp
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from queue import LifoQueue


Decode2Play = LifoQueue()
Decode2Save = LifoQueue()
cameraIndex = 0


class cvDecode(QThread):
    fps_ = 10
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.cap = cv2.VideoCapture(cameraIndex)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    def run(self):
        # print("当前线程cvDecode: self.threadFlag:{}".format(self.threadFlag))
        detector = HandDetector(maxHands=2, detectionCon=0.8)

        x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
        y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        coff = np.polyfit(x, y, 2)
        pTime = 0

        site = np.zeros([1, 164])  # 使用数据时要将第一行数据删除掉
        widght = int(self.cap.get(3))  # 在视频流的帧的宽度,3为编号，不能改
        height = int(self.cap.get(4))  # 在视频流的帧的高度,4为编号，不能改


        while self.threadFlag:
            if self.cap.isOpened():
                ret, r1 = self.cap.read()
                #img = cv2.GaussianBlur(r1, (5, 5), 0, 0)
                img = cv2.flip(r1, 1)
                img1 = cv2.flip(r1, 1)

                # Hands
                hands, img = detector.findHands(img)
                data = []
                lmList = []
                lmList1 = []
                HandLeft = []
                HandRight = []
                # Landmark values
                if hands:
                    hand = hands[0]  # get first hand detected
                    handType = hand['type']  # Get the hand type
                    lmList.append(handType)
                    lmList.append(hand['lmList'])  # Get the landmark list

                    if lmList[0] == "Left" and len(hands) == 1:
                        HandLeft = hand['lmList']
                    else:
                        HandRight = hand['lmList']

                    if len(hands) == 2:
                        hand1 = hands[1]
                        hand1Type = hand1['type']
                        lmList1.append(hand1Type)
                        lmList1.append(hand1['lmList'])

                        if lmList1[0] == "Left":
                            HandLeft = hand1['lmList']
                            HandRight = hand['lmList']
                        else:
                            HandLeft = hand['lmList']
                            HandRight = hand1['lmList']

                    if len(HandLeft) != 0 and len(HandRight) != 0:
                        Lx1, Ly1 = HandLeft[5][:2]
                        Lx2, Ly2 = HandLeft[17][:2]
                        Ldistance = int(math.sqrt((Ly2 - Ly1) ** 2 + (Lx2 - Lx1) ** 2))
                        A, B, C = coff
                        LdistanceCM = A * Ldistance ** 2 + B * Ldistance + C

                        Rx1, Ry1 = HandRight[5][:2]
                        Rx2, Ry2 = HandRight[17][:2]
                        Rdistance = int(math.sqrt((Ry2 - Ry1) ** 2 + (Rx2 - Rx1) ** 2))
                        A, B, C = coff
                        RdistanceCM = A * Rdistance ** 2 + B * Rdistance + C
                    elif len(HandLeft) != 0 and len(HandRight) == 0:
                        Lx1, Ly1 = HandLeft[5][:2]
                        Lx2, Ly2 = HandLeft[17][:2]
                        Ldistance = int(math.sqrt((Ly2 - Ly1) ** 2 + (Lx2 - Lx1) ** 2))
                        A, B, C = coff
                        LdistanceCM = A * Ldistance ** 2 + B * Ldistance + C
                    elif len(HandLeft) == 0 and len(HandRight) != 0:
                        Rx1, Ry1 = HandRight[5][:2]
                        Rx2, Ry2 = HandRight[17][:2]
                        Rdistance = int(math.sqrt((Ry2 - Ry1) ** 2 + (Rx2 - Rx1) ** 2))
                        A, B, C = coff
                        RdistanceCM = A * Rdistance ** 2 + B * Rdistance + C
                    # print(LdistanceCM, RdistanceCM)

                    if len(HandLeft) != 0:
                        data.extend(["Left"])
                    for lm in HandLeft:
                        data.extend([lm[0], height - lm[1], lm[2], LdistanceCM])
                    if len(HandRight) != 0:
                        data.extend(["Right"])
                    for lm1 in HandRight:
                        data.extend([lm1[0], height - lm1[1], lm1[2], RdistanceCM])
                    print(data)
                    site = np.append(site, data)

                np.savetxt('test001', site, delimiter=',', fmt='%s')
                # 算手部追踪的帧率
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                self.fps_ = fps
                cv2.putText(img, f'fps  : {int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                time.sleep(0.01)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                if ret:
                    Decode2Play.put(img)  # 解码后的数据放到队列中
                    Decode2Save.put(img1)  #将原始视频数据放入保存队列中
                del img
                del img1


class play_Work(QThread):  #在UI界面中输出识别后的画面
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化QLabel对象
        # cv2.namedWindow("test")
        # cv2.resizeWindow("test", 640, 480)

    def run(self):
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()

                #frame = cv2.resize(frame, (800, 600), cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示
                # cv2.imshow('test', frame)

            time.sleep(0.001)

class video_Save(QThread): #将原视频保存
    def __init__(self):
        super(video_Save, self).__init__()
        self.threadFlag = 0

    def run(self):
        time.sleep(0.5)
        fps1 = cvDecode.fps_
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 为视频编码方式，保存为mp4文件
        out = cv2.VideoWriter()
        # 定义一个视频存储对象，以及视频编码方式,帧率，视频大小格式
        out.open("video.mp4", fourcc, fps1, (1280, 720))

        while self.threadFlag:
            if not Decode2Save.empty():
                frame = Decode2Save.get()
                while not Decode2Save.empty():
                    Decode2Save.get()
                out.write(frame)
            time.sleep(0.001)





class NameController(QWidget, Ui_Form):

    def __init__(self):
        super().__init__()
        self.setupUi(self)			# 初始化时构建窗口
        self.controller()

    def controller(self):
        self.insert_camera.clicked.connect(lambda: self.insert_video_start())
        self.out_camera.clicked.connect(lambda: self.out_video_start())
        self.videoname.clicked.connect(lambda: self.videoname_start())
        self.videonametext.returnPressed.connect(lambda: self.videoname_start())  #输入文本后按下回车即可进行识别，与点击下方按钮效果相同

    def insert_video_start(self):
        global cameraIndex
        cameraIndex = 0
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

        self.savework = video_Save()
        self.savework.threadFlag = 1
        self.savework.start()



    def out_video_start(self):
        global cameraIndex
        cameraIndex = 1
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

    def videoname_start(self):
        global cameraIndex
        cameraIndex = self.videonametext.text()
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()

        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()


    def closeEvent(self, event):
        print("关闭线程")
        # Qt需要先退出循环才能关闭线程
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
        if self.savework.isRunning():
            self.savework.threadFlag = 0
            self.savework.quit()




if __name__ == '__main__':
    app = QApplication(sys.argv)    # 实例化一个 app
    window = NameController()       # 实例化一个窗口
    window.show()                   # 以默认大小显示窗口


sys.exit(app.exec_())