import cv2
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets


class PoseLandmarkViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("姿态识别")
        self.resize(900, 680)

        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        self.video_label.setStyleSheet("background-color: #000;")

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_capture)

        self.status_label = QtWidgets.QLabel("点击开始查看姿态关键点", alignment=QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.status_label)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.cap = None
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.current_frame = None

    def start_capture(self):
        if self.cap is None or not self.cap_is_open():
            self.cap = self.open_camera()
            if self.cap is None or not self.cap_is_open():
                self.status_label.setText("无法打开摄像头，请检查设备权限或连接")
                return
        if not self.timer.isActive():
            self.timer.start(30)
        self.status_label.setText("识别中...")

    def update_frame(self):
        if self.cap is None or not self.cap_is_open():
            self.status_label.setText("无法打开摄像头")
            self.timer.stop()
            return

        success, frame_bgr = self.cap.read()
        if not success:
            self.status_label.setText("无法读取摄像头画面")
            return

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb.copy())

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(frame_bgr, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

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

    def cap_is_open(self):
        return self.cap is not None and self.cap.isOpened()

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap_is_open():
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
    window = PoseLandmarkViewer()
    window.show()
    app.exec_()

