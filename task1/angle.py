import argparse
import math

# 输入一个整数(高度)，输出对应高度的等腰三角形。要求:1.使用argparse 来输入参数,不能使用input()函数
# 高度小于10，默认是5
# eg
#     1
#    222
#   33333
#  4444444
# 555555555


def triangle(height):
    for i in range(height):
        # print(" " * (height - i - 1) + "*" * (2 * i + 1))
        print(" " * (height - i - 1), end="")
        for j in range(2 * i + 1):
            print(i + 1, end="")
        print()
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="triangle")
    parser.add_argument("--height", type=int, default=5, help="height of triangle")
    args = parser.parse_args()
    triangle(args.height)

