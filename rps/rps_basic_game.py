import random
import time
from typing import Optional

import cv2
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets


class RPSBasicGame(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手势识别 - 剪刀石头布")
        self.resize(900, 680)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet("background-color: #000;")

        self.info_label = QtWidgets.QLabel("点击开始进行游戏", alignment=QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-size: 20px;")

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_game)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.info_label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.cap: Optional[cv2.VideoCapture] = None
        self.score = 0
        self.time_limit = 10
        self.start_time: Optional[float] = None
        self.current_gesture: Optional[str] = None
        self.gestures = ["剪刀", "石头", "布"]
        self.prev_time = None
        self.current_frame = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def start_game(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = self.open_camera()
            if self.cap is None or not self.cap.isOpened():
                self.info_label.setText("无法打开摄像头，请检查设备权限或连接")
                return

        self.score = 0
        self.start_time = time.time()
        self.current_gesture = random.choice(self.gestures)
        self.info_label.setText(f"当前目标：{self.current_gesture}")
        self.prev_time = time.time()

        if not self.timer.isActive():
            self.timer.start(30)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.info_label.setText("无法打开摄像头")
            self.timer.stop()
            return

        success, frame = self.cap.read()
        if not success:
            self.info_label.setText("无法读取摄像头画面")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb.copy())

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_tips = []
                h, w, _ = frame.shape
                for idx, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx in [4, 8, 12, 16, 20]:
                        finger_tips.append((cx, cy, lm.z))
                        cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if len(finger_tips) == 5:
                    user_gesture = self.check_gesture(finger_tips)
                    if user_gesture and self.current_gesture:
                        winner = self.judge_winner(self.current_gesture, user_gesture)
                        if winner == "user":
                            self.score += 1
                        cv2.putText(frame, f"{self.current_gesture} vs {user_gesture}", (10, 100),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                        cv2.putText(frame, f"{winner} wins!", (10, 150),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                        self.current_gesture = random.choice(self.gestures)
                        self.info_label.setText(f"当前目标：{self.current_gesture} | 得分：{self.score}")

        current_time = time.time()
        fps = 0
        if self.prev_time:
            delta = current_time - self.prev_time
            if delta > 0:
                fps = int(1 / delta)
        self.prev_time = current_time

        if fps:
            cv2.putText(frame, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 255, 255), 2)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = display_frame.copy()
        self.update_video_label()

        if self.start_time is not None and (current_time - self.start_time) >= self.time_limit:
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.info_label.setText(f"游戏结束，你的成绩是 {self.score} 分")

    @staticmethod
    def to_qimage(frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
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
    def check_gesture(finger_tips):
        if len(finger_tips) < 5:
            return None
        index_finger = finger_tips[1]
        middle_finger = finger_tips[2]
        ring_finger = finger_tips[3]
        pinky_finger = finger_tips[4]
        thumb_finger = finger_tips[0]
        if index_finger[2] < middle_finger[2] and index_finger[2] < ring_finger[2] and index_finger[2] < pinky_finger[2]:
            if middle_finger[2] < ring_finger[2] and middle_finger[2] < pinky_finger[2]:
                if ring_finger[1] > middle_finger[1] and pinky_finger[1] > middle_finger[1]:
                    if thumb_finger[0] < index_finger[0]:
                        return "剪刀"
        elif index_finger[1] > middle_finger[1] and middle_finger[1] > ring_finger[1] and ring_finger[1] > pinky_finger[1]:
            if thumb_finger[0] < index_finger[0]:
                return "石头"
        elif index_finger[2] < thumb_finger[2]:
            if middle_finger[2] < thumb_finger[2]:
                if ring_finger[2] < thumb_finger[2]:
                    if pinky_finger[2] < thumb_finger[2]:
                        if thumb_finger[0] > index_finger[0]:
                            return "布"
        return None

    @staticmethod
    def judge_winner(computer_gesture, user_gesture):
        if computer_gesture == user_gesture:
            return "tie"
        if computer_gesture == "剪刀":
            return "user" if user_gesture == "石头" else "computer"
        if computer_gesture == "石头":
            return "user" if user_gesture == "布" else "computer"
        if computer_gesture == "布":
            return "user" if user_gesture == "剪刀" else "computer"
        return "unknown"

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
    window = RPSBasicGame()
    window.show()
    app.exec_()
