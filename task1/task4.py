import math
import time
import argparse
import numpy as np
import cv2

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
    task4()