import os
import cv2
import sys
import time
import shutil
import datetime
import json
import numpy as np
from PIL import Image
# from skimage import exposure


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def count_time_seconds(start, end):
    start = datetime.datetime.fromtimestamp(start)
    end = datetime.datetime.fromtimestamp(end)
    return (end - start).seconds


def make_json(contents, save_json):
    json_str = json.dumps(contents, indent=4)
    with open(save_json, "w", encoding='utf-8') as f:
        f.write(json_str)


def read_json(json_path):
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    return class_indict


def read_class_json(json_path):
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    class_name = {int(k): d for k, d in class_indict.items()}

    return class_name


def change_light(image_array, light=0.5):
    image_light_array = exposure.adjust_gamma(image_array, light)
    return image_light_array


def get_image_array(image_path):
    img = cv2.imread(image_path)
    img_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_array


def image2str(img_array):
    line_list = []
    img_list = img_array.tolist()
    for line in img_list:
        for i in range(len(line)):
            line[i] = str(line[i])

    for line in img_list:
        line_str = ','.join(line)
        line_list.append(line_str)

    image_str = ';'.join(line_list)
    return image_str


def str2image(image_str):
    image_list = []
    line_list = image_str.split(';')
    for line in line_list:
        image_list.append(line.split(','))

    for line in image_list:
        for i in range(len(line)):
            line[i] = int(float(line[i]))

    return np.array(image_list, dtype='uint8')


def cuboid2str(cuboid):
    matrix_list = []
    for matrix in cuboid:
        matrix_str = image2str(matrix)
        matrix_list.append(matrix_str)

    return '/'.join(matrix_list)


def str2cuboid(cuboid_str):
    matrix_list = []
    matrix_str_list = cuboid_str.split('/')
    for matrix_str in matrix_str_list:
        matrix_list.append(str2image(matrix_str))

    return np.array(matrix_list, dtype='uint8')


def cuboid2txt(cuboid, cuboid_txt):
    matrix_str_list = []
    for matrix in cuboid:
        matrix_str_list.append(image2str(matrix))

    # print(matrix_str_list)
    f = open(cuboid_txt, 'w', encoding='utf-8')
    for matrix_str in matrix_str_list:
        f.writelines(matrix_str + '\n')
    f.close()


def txt2cuboid(cuboid_txt):
    matrix_array = []
    f = open(cuboid_txt, 'r', encoding='utf-8')
    matrix_list = f.readlines()
    f.close()

    for line in matrix_list:
        matrix_str = line.strip()
        matrix_array.append(str2image(matrix_str))

    return np.array(matrix_array)


def get_analog_one_png(crop_png_path):
    img_array = get_image_array(crop_png_path)
    image_str = image2str(img_array)

    new_image_array = str2image(image_str)
    imgarray = cv2.cvtColor(new_image_array, cv2.COLOR_GRAY2BGR)

    return imgarray


def get_analog_crop_png(crop_png_dir):
    img_rgb_list = []
    crop_imgs = os.listdir(crop_png_dir)

    for crop_img in crop_imgs:
        crop_png_path = os.path.join(crop_png_dir, crop_img)
        img = cv2.imread(crop_png_path, 0)
        # b,g,r = cv2.split(img)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb_list.append(img)

    img_rgb_array = np.array(img_rgb_list, dtype='uint8')
    return img_rgb_array


def clear_txt(txt_path):
    if os.path.isfile(txt_path):
        with open(txt_path, 'r+') as file:
            file.truncate(0)


# 将长方体存入txt
def DI_cuboid_txt(cuboid):
    file = open("python_txt.txt", 'w', encoding='utf-8')
    now_time = datetime.datetime.now()
    now_time_str = datetime.datetime.strftime(now_time, '%Y-%m-%d   %H:%M:%S  ')
    file.writelines(now_time_str + "\n")
    file.writelines("cuboid len: " + str(len(cuboid)) + '\n')
    data = cuboid
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()


def get_txt_lines(txt_path):
    f = open(txt_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    # print(len(lines))
    return len(lines)

# add 2022/09/03, 规范图片命名
def get_image_name_2(num):
    str_num = str(num)
    len_i = len(str_num)
    if len_i==1:
        name = '100'+str_num
    elif len_i==2:
        name = '10'+str_num
    elif len_i==3:
        name = '1'+str_num
    return name

# 将长方体还原成图片image， modify 2022/09/03, 命名规范
def DI_cuboid_pic(cuboid, cv2_dir):
   # cv2_dir = './result_test_1'
    if os.path.exists(cv2_dir):
        shutil.rmtree(cv2_dir)
        os.mkdir(cv2_dir)
    else:
        os.mkdir(cv2_dir)
    count = 0 
    for matrix in cuboid:
        # matrix_path = r"E:\PythonProject\CppProject\DI_log\cv2_test\{}.png".format(count + 1)
        matrix_path = os.path.join(cv2_dir, "{}.png".format(get_image_name_2(count + 1)))
        print(matrix_path)
        cv2.imwrite(matrix_path, matrix, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count += 1


def normalize_hu(image):
    '''
           将输入图像的像素值(-4000 ~ 4000)归一化到0~255之间
       :param image 输入的图像数组
       :return: 归一化处理后的图像数组
    '''

    MIN_BOUND = -1000.0
    MAX_BOUND = 600.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    image = image * 255
    return image


# 长方体预处理
def DI_cuboid_process(cuboid, length, width):
    fo = open("E:\\PythonProject\\CppProject\\DI_log\\foo3.txt", "w")
    new_cuboid_list = []
    img_grey_array = np.array(cuboid)
    img_grey_array = np.reshape(img_grey_array, (-1, length, width))
    len_imgs = len(img_grey_array)
    count = 1

    for img_grey in img_grey_array:
        for i in range(length):
            for j in range(width):
                fo.write(str(img_grey[i][j]))
                fo.write(" ")
    fo.close()

    for i in range(len_imgs):
        img = img_grey_array[i]
        img = normalize_hu(img)  # 阈值转变 deepinsight转换成正常的
        img = Image.fromarray(img).convert("L")
        # length, width = img.size

        # img = trans.resize(img, (64, 64))
        size = max(length, width)
        background = Image.new('L', (size, size), 0)  # 创建黑色背景图
        frame = int(abs(length - width) // 2)  # 一侧需要填充的长度
        box = (frame, 0) if length < width else (0, frame)  # 粘贴的位置
        background.paste(img, box)
        # background.show()
        image_data = background.resize((64, 64))  # 缩放
        image_data = np.array(image_data)

        # img_rgb = cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
        new_cuboid_list.append(image_data)

    new_cuboid = np.array(new_cuboid_list, dtype='uint8')
    DI_cuboid_pic(new_cuboid)
    cuboid2txt(new_cuboid, "cuboid_txt.txt")  # 将长方体保存成txt

    return get_txt_lines("cuboid_txt.txt")


def DI_cuboid_process2():  # filename为写入CSV文件的路径，data为要写入数据列表.
    file = open("python_txt.txt", 'w', encoding='utf-8')
    image_path = r"F:\dataset\Lung-nodules\new_data\crop64_enhance\train\ground_benign\236496"
    cuboid = get_analog_crop_png(image_path)
    data = cuboid
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()


if __name__ == '__main__':
    image_path = r"F:\dataset\Lung-nodules\new_data\crop64_enhance\train\ground_benign\236496"
    # cuboid = get_analog_crop_png(image_path)
    cuboid = get_analog_crop_png(image_path)

    DI_cuboid_pic(cuboid)
    # DI_cuboid_process(cuboid, 20, 20)

    # cuboid2txt(cuboid, "cuboid_txt.txt")
    # res = txt2cuboid("cuboid_txt.txt")
    # print(res)

    # line = get_txt_lines("cuboid_txt.txt")
    # print(line)

    # matrix_array = txt2cuboid("cuboid_txt2.txt")
    # print(matrix_array)

    # cuboid_str = cuboid2str(cuboid)
    #
    # print("cuboid_str:", cuboid_str)
    # print()
    # matrix_array = str2cuboid(cuboid_str)
    # print(matrix_array)
