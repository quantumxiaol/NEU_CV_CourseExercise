import cv2
import mediapipe as mp
import time

import sys
import tkinter as tk


import cv2
import mediapipe as mp
import time


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

if __name__ == '__main__':
    windows = tk.Tk()
    windows.title("手势识别")
    width = 800
    height = 600

    windows.resizable(0, 0)
    t0 = int(time.time())
    t1 = t0   
    #　创建菜单栏
    menubar = tk.Menu(windows)

    filemenu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='选项', menu=filemenu)
    filemenu.add_command(label='内置摄像头', command=invideo, accelerator='F1')
    filemenu.add_command(label='外置摄像头', command=outvideo, accelerator='F2')
    filemenu.add_separator()
    filemenu.add_command(label='退出', command=windows.quit, accelerator='F3')
     
    windows.config(menu=menubar)
    # end 创建菜单栏

    # 创建状态栏
    label = tk.Label(windows, text="手势识别", bd=1, anchor='w')  
    label.pack(side="bottom", fill='x')
    # set_label_text()

    canvas = tk.Canvas(windows, background="#F2F2F2", width = width, height = height)
    canvas.pack()


    windows.mainloop()

