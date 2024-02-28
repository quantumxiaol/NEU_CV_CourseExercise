**剪刀石头布

一、设计目的

本课程设计的目的是利用 Python 语言和 OpenCV 库及Mediapipe库，实现一个基于摄像头的视频捕捉和手势识别的应用程序。该程序可以通过摄像头获取用户的手势，并与电脑进行石头剪刀布的游戏，显示用户和电脑的出招和结果。

二、设计原理

本课程设计主要使用了以下几个技术：

视频捕捉：使用 cv2.VideoCapture(0) 创建一个 VideoCapture 对象，打开摄像头设备，并使用 cap.read() 方法循环读取视频帧。

手势识别：使用 mp.solutions.hands.Hands() 创建一个 mpHands.Hands 对象，用于处理手势识别，并使用 hands.process(image) 方法将图像传入手部检测对象，并获取检测结果。检测结果中包含了多只手的关键点坐标和标签信息，可以用于判断手势的类型。

手势绘制：使用 mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) 方法，在图像上绘制手部关键点和连线，以便观察手势的形状。

帧率计算：使用 time.time() 方法获取当前时间，并与上一帧的时间相减，得到每帧的时间间隔，然后取倒数，得到每秒的帧数（fps）。

图像显示：使用 cv2.putText(image, text, position, font, size, color, thickness) 方法，在图像上绘制文本，显示用户和电脑的出招、结果和帧率等信息。使用 cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 方法将图像转换为 RGB 格式，并使用 tk.PhotoImage(data=cv2.imencode('.png', image)[1].tobytes()) 方法将图像转换为 PhotoImage 对象，以便 Canvas 控件显示。使用 canvas.create_image(0, 0, image=photo, anchor=tk.NW) 方法，在 Canvas 控件上创建一个图像对象，并使用 canvas.update() 方法更新 Canvas 控件。

三、实现方法

1.计算手势的方法

定义一个函数 vector_2d_angle(v1,v2)，用于求解二维向量的角度。该函数接收两个参数 v1 和 v2，分别表示两个二维向量的坐标。该函数使用 math.acos() 方法计算两个向量的夹角的余弦值，并使用 math.degrees() 方法将其转换为角度值。该函数返回一个浮点数表示角度值。如果计算出错或角度大于 180 度，则返回 65535.

定义一个函数 hand_angle(hand_)，用于获取对应手相关向量的二维角度。该函数接收一个参数 hand_，表示一只手的 21 个关键点的坐标列表。该函数使用 vector_2d_angle(v1,v2) 函数计算每个手指（大拇指、食指、中指、无名指、小拇指）的弯曲程度（即相邻两段手指骨之间的夹角）。该函数返回一个包含 5 个元素的列表 angle_list，表示每个手指的弯曲角度值。

定义一个函数 h_gesture(angle_list)，用于根据角度确定手势。该函数接收一个参数 angle_list，表示一只手的 5 个手指的弯曲角度列表。该函数使用一些阈值（thr_angle、thr_angle_thumb、thr_angle_s）来判断每个手指是否张开或闭合，并根据不同的组合来确定手势的类型。该函数返回一个字符串 gesture_str，表示手势的类型。

2.可视化界面的实现方法

使用 cv2、tkinter、mediapipe、random 等模块，定义一个回调函数 start()。通过 VideoCapture 对象捕捉摄像头视频，并循环读取每一帧。通过 mpHands.Hands 对象识别手部关键点，并通过 mp.solutions.drawing_utils 对象绘制手势。通过计算手指的弯曲角度，判断手势的类型，并与电脑进行石头剪刀布的游戏。通过 cv2.putText() 方法显示帧率、出招和结果等信息。通过 tk.PhotoImage() 方法将图像转换为 Canvas 控件可用的格式，并在 Canvas 控件上显示图像。通过 tkinter 模块创建一个窗口，并添加一个 Canvas 控件和一个 Button 控件，将 start() 函数绑定到 Button 控件上。

启动 tkinter 主循环。可视化搭建完成
