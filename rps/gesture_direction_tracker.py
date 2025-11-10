import math
from typing import Optional

import cv2
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets


class GestureDirectionTracker(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手指方向识别")
        self.resize(900, 680)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet("background-color: #000;")

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_tracking)

        self.status_label = QtWidgets.QLabel("点击开始识别方向", alignment=QtCore.Qt.AlignCenter)
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

        self.directions = {
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT"
        }
        self.direction_cache = None
        self.direction_count = 0

    def start_tracking(self):
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

        success, frame = self.cap.read()
        if not success:
            self.status_label.setText("无法读取摄像头画面")
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb.copy())
        detected_direction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                y1 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                x2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                y2 = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                if x2 == x1:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                angle = math.degrees(math.atan(slope))

                if (angle > 45 or angle < -45) and y1 > y2:
                    detected_direction = self.directions["up"]
                elif (angle > 45 or angle < -45) and y1 < y2:
                    detected_direction = self.directions["down"]
                elif -45 < angle < 45 and x1 < x2:
                    detected_direction = self.directions["right"]
                else:
                    detected_direction = self.directions["left"]

                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                stable_direction = self.counts(detected_direction)
                if stable_direction:
                    cv2.putText(frame, stable_direction, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                    self.status_label.setText(f"当前方向：{stable_direction}")

        display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = display_image.copy()
        self.update_video_label()

    def counts(self, direction, count=10):
        if direction is None:
            self.direction_cache = None
            self.direction_count = 0
            return None
        if self.direction_cache != direction:
            self.direction_cache = direction
            self.direction_count = 1
        else:
            self.direction_count += 1
            if self.direction_count >= count:
                self.direction_count = 0
                return direction
        return None

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
    window = GestureDirectionTracker()
    window.show()
    app.exec_()







    
 



