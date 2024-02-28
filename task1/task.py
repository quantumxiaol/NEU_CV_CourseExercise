import math
import time
import argparse
import numpy as np
import cv2

# 【实验1-1】用Python实现斐波那契序列。提示:使用input()输入项数
# n = int(input("请输入要求斐波那契数列前几项:"))

def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()
    return 0

def task1(n):
    print("斐波那契数列前%d项为:" % n)
    fib(n)
    return 0

# 【实验1-2】用Python实现两个数组中对应元素相乘并累加，即点积运算。用for循环实现，给出计算结果，并给出这段程序的运行时长。
# 提示:使用time.time()函数计算时间
def dot_product(a, b):
    # 获取以秒为单位的当前时间可使用time.perf_counter()函数
    # time_start = time.time()
    time_start = time.perf_counter()
    sum = 0
    for i in range(len(a)):
        sum += a[i] * b[i]
    time_end = time.perf_counter()
    print("运行时长为: %f 秒" % (time_end - time_start))
    
    return sum

def numpy_dot_product(a, b):
    time_start = time.perf_counter()
    sum = np.dot(a, b)
    time_end = time.perf_counter()
    print("运行时长为: %f 秒" % (time_end - time_start))
    return sum

# n=1000
# a=[1,2....999]
# b=[1000,1001,....1999]
n = 1000
a = []
b = []

def task2(a, b):
    for i in range(n):
        a.append(i)
        b.append(i+n)
    print("点积运算结果为：", dot_product(a, b))
    return 0

a_np = np.array(a)
b_np = np.array(b)

def task3(a_np, b_np):
    print("点积运算结果为：", numpy_dot_product(a_np, b_np))
    return 0

def rotate(img, angle):
    # 获取图像的高和宽
    h, w = img.shape[:2]
    # 计算图像中心点
    center = (w // 2, h // 2)
    # 获得旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 仿射变换
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def mirror(img, mode):
    # mode: 0:左右镜像 1:上下镜像
    if mode == 0:
        return cv2.flip(img, 1)
    elif mode == 1:
        return cv2.flip(img, 0)
    else:
        return img
    
def brighten(img, alpha, beta):
    # alpha: 亮度增益
    # beta: 亮度偏移
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def add_noise(img, mode):
    # mode: 0:高斯噪声 1:椒盐噪声
    if mode == 0:
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif mode == 1:
        return cv2.medianBlur(img, 5)
    else:
        return img
    
def task4():
    img = cv2.imread("img.png")
    img_rotate = rotate(img, 45)
    cv2.imwrite("img_rotate.png", img_rotate)
    img_mirror_0 = mirror(img, 0)
    cv2.imwrite("img_lr.png", img_mirror_0)
    img_mirror_1 = mirror(img, 1)
    cv2.imwrite("img_ud.png", img_mirror_1)
    img_brighten = brighten(img, 1.5, 0)
    cv2.imwrite("img_lighte.png", img_brighten)
    img_noise = add_noise(img, 0)
    cv2.imwrite("img_noise.png", img_noise)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triangle")
    parser.add_argument("--num", type=int, default=5, help="height of triangle")
    args = parser.parse_args()
    task1(args.num)
    task2(a, b)
    task3(a_np, b_np)
    task4()