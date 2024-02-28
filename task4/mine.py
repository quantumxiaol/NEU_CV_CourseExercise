import tkinter as tk
import cv2
import mediapipe as mp
import time
import random

root = tk.Tk()
root.title("手势识别")
root.geometry("800x600+100+100")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

button = tk.Button(root, text="Start", command=lambda: start())
button.pack()

label = tk.Label(root, text="", font=("Arial", 20))
label.pack()

score = 0 # 积分
time_limit = 10 # 时间限制
start_time = None # 开始时间

gestures = ["剪刀", "石头", "布"] # 可能的手势
current_gesture = None # 当前的手势

def start():
    global start_time, current_gesture
    start_time = time.time() # 记录开始时间
    current_gesture = random.choice(gestures) # 随机选择一个手势
    label.config(text=current_gesture) # 显示当前的手势
    update() # 更新界面

def update():
    global score, time_limit, start_time, current_gesture
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    pTime = 0
    cTime = 0
    success, img = cap.read()
    if not success:
        return
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            finger_tips = [] # 存储手指尖的位置和状态
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8, 12, 16, 20]: # 手指尖的id
                    finger_tips.append((cx, cy, lm.z)) # 添加位置和深度信息
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) # 在手指尖画圆
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            user_gesture = check_gesture(finger_tips) # 检测用户的手势
            if user_gesture: # 如果用户做出了手势
                winner = judge_winner(current_gesture, user_gesture) # 判断获胜方
                if winner == "user": # 如果用户赢了
                    score += 1 # 积分加一
                cv2.putText(img, f"{current_gesture} vs {user_gesture}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 255, 255), 2) # 在屏幕上显示两个手势
                cv2.putText(img, f"{winner} wins!", (10, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 255, 255), 2) # 在屏幕上显示获胜方
                current_gesture = random.choice(gestures) # 随机选择下一个手势
                label.config(text=current_gesture) # 显示下一个手势

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 255, 255), 2)

    photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    
    elapsed_time = cTime - start_time # 计算已经过去的时间
    if elapsed_time < time_limit: # 如果还没有超过时间限制
        canvas.after(10, update) # 继续更新界面
    else: # 如果超过了时间限制
        cap.release() # 释放摄像头
        label.config(text=f"游戏结束，你的成绩是{score}分") # 显示最终的成绩

def check_gesture(finger_tips):
    # 根据手指尖的位置和状态判断用户的手势
    # finger_tips是一个列表，每个元素是一个元组，包含了手指尖的x坐标，y坐标和z深度
    # 返回一个字符串，表示用户的手势，如果没有做出手势，返回None
    index_finger = finger_tips[1] # 食指
    middle_finger = finger_tips[2] # 中指
    ring_finger = finger_tips[3] # 无名指
    pinky_finger = finger_tips[4] # 小拇指
    thumb_finger = finger_tips[0] # 拇指
    if index_finger[2] < middle_finger[2] and index_finger[2] < ring_finger[2] and index_finger[2] < pinky_finger[2]:
        # 如果食指的深度小于其他三个手指，说明食指伸直了
        if middle_finger[2] < ring_finger[2] and middle_finger[2] < pinky_finger[2]:
            # 如果中指的深度小于无名指和小拇指，说明中指也伸直了
            if ring_finger[1] > middle_finger[1] and pinky_finger[1] > middle_finger[1]:
                # 如果无名指和小拇指的y坐标大于中指，说明它们弯曲了
                if thumb_finger[0] < index_finger[0]:
                    # 如果拇指的x坐标小于食指，说明拇指也弯曲了
                    return "剪刀" # 完成了剪刀手势
    elif index_finger[1] > middle_finger[1] and middle_finger[1] > ring_finger[1] and ring_finger[1] > pinky_finger[1]:
        # 如果食指的y坐标大于中指，中指大于无名指，无名指大于小拇指，说明这四个手指都弯曲了
        if thumb_finger[0] < index_finger[0]:
            # 如果拇指的x坐标小于食指，说明拇指也弯曲了
            return "石头" # 完成了石头手势
    elif index_finger[2] < thumb_finger[2]:
        # 如果食指的深度小于拇指，说明食指伸直了
        if middle_finger[2] < thumb_finger[2]:
            # 如果中指的深度小于拇指，说明中指伸直了
            if ring_finger[2] < thumb_finger[2]:
                # 如果无名指的深度小于拇指，说明无名指伸直了
                if pinky_finger[2] < thumb_finger[2]:
                    # 如果小拇指的深度小于拇指，说明小拇指伸直了
                    if thumb_finger[0] > index_finger[0]:
                        # 如果拇指的x坐标大于食指，说明拇指也伸直了
                        return "布" # 完成了布手势
    return None # 没有完成任何手势

def judge_winner(computer_gesture, user_gesture):
    # 根据剪刀石头布的规则判断获胜方
    # computer_gesture是一个字符串，表示电脑随机选择的手势
    # user_gesture是一个字符串，表示用户做出的手势
    # 返回一个字符串，表示获胜方，如果平局，返回"tie"
    if computer_gesture == user_gesture:
        return "tie" # 平局
    elif computer_gesture == "剪刀":
        if user_gesture == "石头":
            return "user" # 用户赢了
        else:
            return "computer" # 电脑赢了
    elif computer_gesture == "石头":
        if user_gesture == "布":
            return "user" # 用户赢了
        else:
            return "computer" # 电脑赢了
    elif computer_gesture == "布":
        if user_gesture == "剪刀":
            return "user" # 用户赢了
        else:
            return "computer" # 电脑赢了

root.mainloop()
