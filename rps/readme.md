# RPS 手势识别练习

该目录包含 4 个基于 PyQt5 + OpenCV + MediaPipe 的小程序，用于摄像头手势/姿态识别演示。所有脚本都复用了统一的摄像头初始化逻辑（优先 `cv2.CAP_AVFOUNDATION`，失败回退默认后端），并通过 `QTimer` 完成非阻塞刷新，结合 `QLabel` + `QImage` 渲染画面。每个窗口在关闭时都会释放 `VideoCapture`，避免摄像头资源被占用。

## 脚本概览

- `rps_basic_game.py`：检测手指尖位置，判断剪刀/石头/布并与电脑轮番对战，展示比分与 FPS。
- `rps_angle_based_game.py`：使用向量夹角判定多种手势，若识别出剪刀/石头/布则立即与电脑对局并显示结果。
- `gesture_direction_tracker.py`：追踪食指方向，带去抖动逻辑（连续帧计数后才显示方向文字）。
- `pose_landmark_viewer.py`：展示全身姿态关键点，方便调试 MediaPipe Pose。

## 核心技术点

- **视频采集**：`cv2.VideoCapture` 持久化在全局，避免每帧重新打开摄像头；窗口关闭时调用 `release()`。
- **MediaPipe 检测**：`mp.solutions.hands.Hands()` 与 `mp.solutions.pose.Pose()` 获取关键点坐标（含左右手标签）。
- **UI 渲染**：OpenCV 图像先转 RGB，再封装为 `QImage`/`QPixmap` 显示在 `QLabel` 上。
- **事件循环**：使用 `QTimer` 定时拉取摄像头帧，避免阻塞 Qt 事件循环。
- **资源清理**：统一提供 `on_close()` 钩子释放摄像头并销毁窗口。

## 运行方式

在 macOS 终端进入项目目录并激活虚拟环境后运行：

```bash
python rps/rps_basic_game.py
python rps/rps_angle_based_game.py
python rps/gesture_direction_tracker.py
python rps/pose_landmark_viewer.py
```

如需避免窗口被 IDE 捕获，可在系统终端使用 `pythonw`。首次运行若摄像头无法打开，请确认终端拥有摄像头权限，并确保摄像头未被其他程序占用。
