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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triangle")
    parser.add_argument("--num", type=int, default=5, help="height of triangle")
    args = parser.parse_args()
    task1(args.num)