# 导入所需的库
import cv2
import mediapipe as mp
import math
import tkinter as tk # 导入tkinter库
from PIL import Image, ImageTk # 导入PIL库
import time

# 创建手部检测对象和绘图对象
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 定义方向字典，用于存储食指的方向和对应的反馈信息
directions = {
    "up": "UP",
    "down": "DOWN",
    "left": "LEFT",
    "right": "RIGHT"
}

# 设定重复帧 减少误差带来的影响
init = 0
direction_ = None
# 创建一个主窗口对象
root = tk.Tk()
# 设置主窗口的标题
root.title("方向识别")
# 设置主窗口的大小和位置
root.geometry("800x600+100+100")

# 创建一个 Canvas 控件，用于显示视频图像
canvas = tk.Canvas(root, width=640, height=480)
# 使用 pack 几何管理器将 Canvas 控件添加到主窗口中
canvas.pack()

# 创建一个 Button 控件，显示 "Start" 文本，并绑定一个回调函数
button = tk.Button(root, text="Start", command=lambda: start())
# 使用 pack 几何管理器将 Button 控件添加到主窗口中
button.pack()
# 只有当相同识别帧数到count帧时, 才会去显示对应的方向
def counts(direction, count=10):
    global init, direction_
    if direction_ != direction:
        direction_ = direction
        init = 0
    elif direction_ == direction:
        init += 1
        if init >= count:
            return direction_
    return None

# 定义一个回调函数，用于启动视频捕捉和手势识别
def start():
    # 创建一个 VideoCapture 对象，打开摄像头设备
    cap = cv2.VideoCapture(0)

    # 创建一个 mpHands.Hands 对象，用于处理手势识别
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 创建一个 mp.solutions.drawing_utils 对象，用于绘制手势标记点和连接线
    mpDraw = mp.solutions.drawing_utils

    # 初始化两个变量，用于计算帧率
    pTime = 0
    cTime = 0

    # 循环读取视频帧，并进行手势识别和绘制
    while True:
        # 读取一帧视频，并判断是否成功
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        # 将图像传入手部检测对象，并获取检测结果
        results = hands.process(image)

        # 如果检测到至少一只手，则遍历每只手，并获取食指的方向和反馈信息
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取食指第二个关节（index_finger_mcp）和第三个关节（index_finger_pip）的坐标
                x1 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                y1 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                x2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x
                y2 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                # 计算食指的斜率和角度（以度为单位）
                slope = (y2 - y1) / (x2 - x1)
                angle = math.atan(slope) * 180 / math.pi

                # 根据角度判断食指的方向，并获取对应的反馈信息
                if (angle > 45 or angle < -45) and y1 > y2:
                    direction = directions["up"]
                elif (angle > 45 or angle < -45) and y1 < y2:
                    direction = directions["down"]
                elif (-45 < angle < 45) and x1 < x2:
                    direction = directions["right"]
                else:
                    direction = directions["left"]

                # 在图像上绘制手部关键点和连线，并显示反馈信息
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if direction == counts(direction):
                    cv2.putText(image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 将图像转换回BGR格式，并转换为tkinter可用的格式（PhotoImage）
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        # image = tk.PhotoImage(image=image)
        # image = Image.fromarray(image)
        # image = IMAGETEXT.PhotoImage(image)
        # image = ImageTk.PhotoImage(image)

        # 在图像上绘制帧率的文本
        # cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 255), 2)

        # 将图像转换为 PhotoImage 对象，以便 Canvas 控件显示
        photo = tk.PhotoImage(data=cv2.imencode('.png', image)[1].tobytes())
        # 在 Canvas 控件上创建一个图像对象，使用 PhotoImage 对象作为源
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        # 更新 Canvas 控件的显示
        canvas.update()
        # 等待按键，如果按下 Esc 键，则退出循环
        key = cv2.waitKey(1)
        if key == 27:
            break

    # 释放 VideoCapture 对象
    cap.release()

# 启动主窗口的消息循环
root.mainloop()







    
 



