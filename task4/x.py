# 导入 tkinter 模块
import tkinter as tk
# 导入 cv2 和 mp.solutions 模块
import cv2
import mediapipe as mp
import time

# 创建一个主窗口对象
root = tk.Tk()
# 设置主窗口的标题
root.title("手势识别")
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
        success, img = cap.read()
        if not success:
            break

        # 将视频帧转换为 RGB 格式，以便 hands.process 方法处理
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 调用 hands.process 方法，传入 RGB 图像，得到识别结果
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks) # 检查手坐标输出

        # 如果识别结果中有多个手的标记点，则遍历每个手的标记点，并绘制出来
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    # 获取图像的高度、宽度和通道数
                    h, w, c = img.shape
                    # 将标记点的相对坐标转换为绝对坐标（像素值）
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    # 在图像上绘制一个圆圈，表示标记点的位置
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                # 在图像上绘制连接线，表示手的轮廓和关节
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # 计算当前时间，用于计算帧率
        cTime = time.time()
        # 计算帧率，即每秒处理的帧数
        fps = 1 / (cTime - pTime)
        # 更新上一次的时间
        pTime = cTime

        # 在图像上绘制帧率的文本
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)

        # 将图像转换为 PhotoImage 对象，以便 Canvas 控件显示
        photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
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

def invideo():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 2 = to
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)//检查手坐标输出
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    # if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    return 0

def outvideo():
    cap = cv2.VideoCapture(1)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 2 = to
        results = hands.process(imgRGB)
        # print(results.multi_hand_landmarks)//检查手坐标输出
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    # if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    return 0