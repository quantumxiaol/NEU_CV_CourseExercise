# 导入所需的库
import cv2
import mediapipe as mp
import math

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


# 打开摄像头并捕获视频帧
cap = cv2.VideoCapture(0)
while cap.isOpened():
    # 读取一帧图像并转换为RGB格式
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



    # 将图像转换回BGR格式，并显示在窗口中
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hand Gesture Recognition", image)

    # 按下q键退出循环并关闭窗口
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows() 