import os
import glob
import numpy as np
# import skimage.io as io
import cv2
from skimage import measure

def label_data_gene(image_arr, image_path) -> list:  #加载图像
    image_name_arr = glob.glob(os.path.join(image_path, "*"))
    for index, item in enumerate(image_name_arr):
        if os.path.isdir(item):
            image_arr = label_data_gene(image_arr, item)
        else:
            img = cv2.imread(item, cv2.IMREAD_GRAYSCALE)
            img = img / 255.0
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img.flatten()
            image_arr.append(img)
    return image_arr

def getDice(seg, gt):
    '''
    评估标准
    Dice=2*(seg ∩ gt)/seg + gt
    OR=Os/Rs+Os
    UR=Us/Rs+Os
    Rs标记图像中的像素点
    Os预测结果中存在但标记图像中不存在的像素点
    Us标记图像中存在但预测结果中不存在的像素点
    :param seg:预测的分割结果像素点
    :param gt:标记图像像素点
    :return: Dice OR UR
    '''
    seg, gt = np.array(seg), np.array(gt)
    smooth = 1  # 防止被零除
    seg_c = seg.flatten()
    gt_c = gt.flatten()  # 矩阵二维化
    intersection = np.sum(seg_c * gt_c)  #seg ∩ gt
    aa = (2. * intersection + smooth)  #2*（seg ∩ gt)
    bb = (np.sum(seg_c) + np.sum(gt_c) + smooth)  #seg+gt
    Dice = aa / bb
    Rs = np.sum(gt_c)
    Os = np.sum(seg_c) - intersection
    Us = Rs - intersection
    OR = (Os + smooth) / (Rs + Os + smooth)
    UR = (Us + smooth) / (Rs + Os + smooth)
    return Dice, OR, UR


def lobe_post_processing(image):
    # 对深度学习分割的lobe进行二值化
    # image[image > 0.1] = 1
    # image[image <= 0.1] = 0
    image = np.reshape(image, (len(image), 512, 512))
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    # 去除面积较小的连通域
    if len(num_list_sorted) > 1:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[1:]:
            # label[label==i] = 0
            if (area_list[i - 1] * 2) < max(area_list):
                # print(i-1)
                label[region[i - 1].slice][region[i - 1].image] = 0
        # num_list_sorted = num_list_sorted[:1]
    label[label > 0] = 1
    return label


# test_path = r'E:\AscendAI\Unet_segment_git\results\nestedunet_adam'    # png 文件
# label_path = r'E:\AscendAI\Unet_segment_git\data\validation\label'  # 灰度图 png文件

# 后处理前：
# test_path = r'E:\PythonProject\DeepLearning\data_statistics\vessel_result'    # png 文件
# label_path = r'E:\PythonProject\DeepLearning\data_statistics\new_label\vessel'  # 灰度图 png文件

#后处理后：
test_path = r'E:\PythonProject\DeepLearning\data_statistics\proprocess_data\vessel'    # png 文件
label_path = r'E:\PythonProject\DeepLearning\data_statistics\proprocess_data\vessel_label'  # 灰度图 png文件


test_dir = test_path
label_dir = label_path
print('test_dir',test_dir, "\n", 'label_dir', label_dir)

# 计算DICE准确率、 过分割率OR、 欠分割率UR
label_data = []
result = []
label_data = label_data_gene(label_data, label_dir)
result = label_data_gene(result, test_dir)
dice, OR, UR = getDice(result, label_data)

print('dice:', round(dice, 4), '        OR:', round(OR, 4), '        UR:', round(UR, 4))

"""""
后处理
"""""
result = lobe_post_processing(result)

# 计算DICE准确率、 过分割率OR、 欠分割率UR
# label_data = []
# label_data = label_data_gene(label_data, label_dir)
dice, OR, UR = getDice(result, label_data)
print('dice:', round(dice, 4), '        OR:', round(OR, 4), '        UR:', round(UR, 4))
