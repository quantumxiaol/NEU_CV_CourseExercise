import math
import time
import argparse
import numpy as np
import cv2
# 【实验1-2】用Python实现两个数组中对应元素相乘并累加，即点积运算。用for循环实现，给出计算结果，并给出这段程序的运行时长。
# 提示:使用time.time()函数计算时间

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
for i in range(n):
    a.append(i)
    b.append(i+n)

a_np = np.array(a)
b_np = np.array(b)

def task3(a_np, b_np):
    print("点积运算结果为：", numpy_dot_product(a_np, b_np))
    return 0

if __name__ == "__main__":
    task3(a, b)