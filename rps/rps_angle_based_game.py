import math
import random
import time
from typing import Optional

import cv2
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets


class AngleBasedRPSGame(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("剪刀石头布 - 角度识别版")
        self.resize(900, 680)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet("background-color: #000;")

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_game)

        self.status_label = QtWidgets.QLabel("点击开始识别手势", alignment=QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.cap: Optional[cv2.VideoCapture] = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.current_frame = None
        self.last_user_gesture = None
        self.last_result_ts = 0.0
        self.result_cooldown = 1.2

    def start_game(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = self.open_camera()
            if self.cap is None or not self.cap.isOpened():
                self.status_label.setText("无法打开摄像头，请检查设备权限或连接")
                return
        if not self.timer.isActive():
            self.timer.start(30)
        self.status_label.setText("识别中...")

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.status_label.setText("无法打开摄像头")
            self.timer.stop()
            return

        success, frame_bgr = self.cap.read()
        if not success:
            self.status_label.setText("无法读取摄像头画面")
            return

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb.copy())
        gesture_str = None

        if results.multi_handedness:
            for hand_label in results.multi_handedness:
                hand_jugg = str(hand_label).split('"')[1]
                cv2.putText(frame_bgr, hand_jugg, (50, 200), 0, 1.3, (0, 0, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame_rgb.shape[1]
                    y = hand_landmarks.landmark[i].y * frame_rgb.shape[0]
                    hand_local.append((x, y))
                if hand_local:
                    angle_list = self.hand_angle(hand_local)
                    gesture_str = self.h_gesture(angle_list)
                    if gesture_str == "0":
                        gesture_str = "ROCK"
                    elif gesture_str == "2":
                        gesture_str = "SCISSORS"
                    elif gesture_str == "5":
                        gesture_str = "PAPER"
                    else:
                        gesture_str = None
                    if gesture_str:
                        cv2.putText(frame_bgr, gesture_str, (50, 100), 0, 1.3, (0, 0, 255), 2)

        current_ts = time.time()

        if gesture_str:
            com_gesture = random.choice(["ROCK", "SCISSORS", "PAPER"])
            if (
                (self.last_user_gesture != gesture_str)
                or (current_ts - self.last_result_ts) >= self.result_cooldown
            ):
                cv2.putText(frame_bgr, com_gesture, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                result = self.win_geasture(gesture_str, com_gesture)
                cv2.putText(frame_bgr, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                self.status_label.setText(f"你：{gesture_str} | 电脑：{com_gesture} | 结果：{result}")
                self.last_user_gesture = gesture_str
                self.last_result_ts = current_ts
        else:
            if (current_ts - self.last_result_ts) >= self.result_cooldown:
                self.status_label.setText("等待识别手势...")

        display_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.current_frame = display_image.copy()
        self.update_video_label()

    @staticmethod
    def to_qimage(frame):
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        return QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

    @staticmethod
    def open_camera():
        cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if cam.isOpened():
            return cam
        cam.release()
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            return cam
        cam.release()
        return None

    @staticmethod
    def vector_2d_angle(v1, v2):
        v1_x, v1_y = v1
        v2_x, v2_y = v2
        try:
            angle_ = math.degrees(math.acos(
                (v1_x * v2_x + v1_y * v2_y) /
                (((v1_x ** 2 + v1_y ** 2) ** 0.5) * ((v2_x ** 2 + v2_y ** 2) ** 0.5))
            ))
        except Exception:
            angle_ = 65535.
        if angle_ > 180.:
            angle_ = 65535.
        return angle_

    def hand_angle(self, hand_):
        angle_list = []
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
            ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
        ))
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
            ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
        ))
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
            ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
        ))
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
            ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
        ))
        angle_list.append(self.vector_2d_angle(
            ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
            ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
        ))
        return angle_list

    @staticmethod
    def h_gesture(angle_list):
        thr_angle = 65.
        thr_angle_thumb = 53.
        thr_angle_s = 49.
        gesture_str = "Unknown"
        if 65535. not in angle_list:
            if (angle_list[0] > thr_angle_thumb) and all(angle_list[i] > thr_angle for i in range(1, 5)):
                gesture_str = "0"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and all(
                    angle_list[i] > thr_angle for i in range(2, 5)):
                gesture_str = "2"
            elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and all(
                    angle_list[i] < thr_angle_s for i in range(2, 5)):
                gesture_str = "4"
            elif all(angle_list[i] < thr_angle_s for i in range(5)):
                gesture_str = "5"
        return gesture_str

    @staticmethod
    def win_geasture(geasture_user, geasture_computer):
        if geasture_user == geasture_computer:
            return "E"
        if geasture_user == "SCISSORS":
            return "Lose" if geasture_computer == "ROCK" else "Win"
        if geasture_user == "ROCK":
            return "Lose" if geasture_computer == "PAPER" else "Win"
        if geasture_user == "PAPER":
            return "Lose" if geasture_computer == "SCISSORS" else "Win"
        return "Unknown"

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)

    def update_video_label(self):
        if self.current_frame is None:
            return
        qt_image = self.to_qimage(self.current_frame)
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.update_video_label()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = AngleBasedRPSGame()
    window.show()
    app.exec_()







    
 



