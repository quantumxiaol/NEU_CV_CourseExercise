import math
import time
import argparse
import numpy as np
import cv2
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

if __name__ == "__main__":
    task2(a, b)