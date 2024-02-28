# 导入所需的库
import cv2
import mediapipe as mp
import math
import tkinter as tk # 导入tkinter库
from PIL import Image, ImageTk , ImageDraw, ImageFont # 导入PIL库
import time
import random

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

geasture = {
    "scissors": "SCISSORS",
    "rock": "ROCK",
    "paper": "PAPER"
}

def vector_2d_angle(v1,v2):
    '''
        求解二维向量的角度
    '''
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_
def hand_angle(hand_):
    '''
        获取对应手相关向量的二维角度,根据角度确定手势
    '''
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 无名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def h_gesture(angle_list):
    '''
        # 二维约束的方法定义手势
        # fist five gun love one six three thumbup yeah
    '''
    thr_angle = 65.  #手指闭合则大于这个值（大拇指除外）
    thr_angle_thumb = 53.  #大拇指闭合则大于这个值
    thr_angle_s = 49.  #手指张开则小于这个值
    gesture_str = "Unknown"
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "0"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "1"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "2"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "3"
        elif (angle_list[0]>thr_angle_thumb) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "4"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "5"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "6"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "8"
            
        elif (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "Pink Up"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "Thumb Up"
        elif (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "Fuck"
        elif (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "Princess"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "Bye"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "Spider-Man"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "Rock'n'Roll"
        
    return gesture_str


def random_geasture():
    return random.choice(list(geasture.values()))

# 剪刀石头布的胜负规则
def win_geasture(geasture_user,geasture_computer):
    if geasture_user == geasture_computer:
        return "E"
    elif geasture_user == "SCISSORS":
        if geasture_computer == "ROCK":
            return "Lose"
        elif geasture_computer == "PAPER":
            return "Win"
    elif geasture_user == "ROCK":
        if geasture_computer == "PAPER":
            return "Lose"
        elif geasture_computer == "SCISSORS":
            return "Win"
    elif geasture_user == "PAPER":
        if geasture_computer == "SCISSORS":
            return "Lose"
        elif geasture_computer == "ROCK":
            return "Win"



# 设定重复帧 减少误差带来的影响
init = 0
direction_ = None
# 创建一个主窗口对象
root = tk.Tk()
# 设置主窗口的标题
root.title("剪刀石头布")
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

# 绘制虚拟小火柴人，显示电脑的出招(剪刀 石头 布)
# def draw_gesture(image, gesture):





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
        gesture_str = None

        # 如果检测到至少一只手，则遍历每只手，并获取食指的方向和反馈信息
        # if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks:
        if results.multi_handedness:
            for hand_label in results.multi_handedness:
                hand_jugg=str(hand_label).split('"')[1]
                print(hand_jugg)
                cv2.putText(image,hand_jugg,(50,200),0,1.3,(0,0,255),2)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*image.shape[1]
                    y = hand_landmarks.landmark[i].y*image.shape[0]
                    hand_local.append((x,y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    # 0 为rock 2 为scissors 5 为paper
                    if gesture_str == "0":
                        gesture_str = "ROCK"
                    elif gesture_str == "2":
                        gesture_str = "SCISSORS"
                    elif gesture_str == "5":
                        gesture_str = "PAPER"
                    else:
                        gesture_str = None


                    # print(gesture_str)
                    cv2.putText(image,gesture_str,(50,100),0,1.3,(0,0,255),2)


                # 在图像上绘制手部关键点和连线，并显示反馈信息
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # if direction == counts(direction):
                #     cv2.putText(image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        # 电脑随机出招，用户出招，判断输赢
        # 未检测到则电脑不出招
        if gesture_str :
            com_gesture = random.choice(["ROCK", "SCISSORS", "PAPER"])
            # 显示电脑的出招
            cv2.putText(image, com_gesture, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
            # 绘制虚拟小火柴人，显示电脑的出招
            # draw_gesture(image, com_gesture, (50, 200), (0,0,255), 1.5, 3)
            


            user_gesture = gesture_str
            result = win_geasture(user_gesture, com_gesture)
            # 显示结果
            cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        else:
            pass

        # cv2.putText(image,gesture_str,(50,100),0,1.3,(0,0,255),2)        

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
        key = cv2.waitKey(60)
        if key == 27:
            break

    # 释放 VideoCapture 对象
    cap.release()

# 启动主窗口的消息循环
root.mainloop()







    
 



