import tkinter as tk
import cv2
import mediapipe as mp
import time

root = tk.Tk()
root.title("姿态识别")
root.geometry("800x600+100+100")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

button = tk.Button(root, text="Start", command=lambda: start())
button.pack()


def start():
    cap = cv2.VideoCapture(0)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # 获取关键点坐标
            landmarks = results.pose_landmarks.landmark

            # 左臂
            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])),
                     (int(landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0])),
                     (255, 0, 0), 3)

            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y * img.shape[0])),
                     (int(landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y * img.shape[0])),
                     (255, 0, 0), 3)

            # 右臂
            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])),
                     (int(landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0])),
                     (255, 0, 0), 3)

            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0])),
                     (int(landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.RIGHT_WRIST.value].y * img.shape[0])),
                     (255, 0, 0), 3)

            # 躯干
            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])),
                     (int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])),
                     (255, 0, 0), 3)

            # 左腿











            
            cv2.line(img,
                     (int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x * img.shape[1]),
                      int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y * img.shape[0])),
                        (int(landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x * img.shape[1]),
                            int(landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y * img.shape[0])),
                        (255, 0, 0), 3)
            
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.LEFT_KNEE.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.LEFT_ANKLE.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            # 右腿
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            # 身体
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.LEFT_HIP.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            # 脖子
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.LEFT_EAR.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.LEFT_EAR.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
            cv2.line(img,
                        (int(landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].x * img.shape[1]),
                        int(landmarks[mpPose.PoseLandmark.RIGHT_EAR.value].y * img.shape[0])),
                            (int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1]),
                                int(landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0])),
                            (255, 0, 0), 3)
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 2)
        photo = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.update()
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
root.mainloop()
            