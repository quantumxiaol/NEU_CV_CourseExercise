import glob
import os
import skimage.io as io
import numpy as np
'''
def getDice(seg, gt):
    seg, gt = np.array(seg), np.array(gt)
    smooth = 1  # 防止被零除
    seg_c = seg.flatten()
    gt_c = gt.flatten()  # 矩阵二维化
    intersection = np.sum(seg_c * gt_c)
    aa = (2. * intersection + smooth)
    bb = (np.sum(seg_c) + np.sum(gt_c) + smooth)
    Dice = aa / bb
    Rs = np.sum(gt_c)
    Os = np.sum(seg_c) - intersection
    Us = Rs - intersection
    OR = (Os + smooth) / (Rs + Os + smooth)
    UR = (Us + smooth) / (Rs + Os + smooth)
    return Dice, OR, UR
'''
def getDice(seg, gt):
    gt = np.array(gt, dtype=np.int32)
    seg = np.array(seg, dtype=np.int32)
    seg = np.array(seg).flatten()
    gt = np.array(gt).flatten()
    label_list = np.unique(gt)

    smooth = 1  # 防止被零除
    intersection = 0
    for i, ele in enumerate(label_list):
        if ele == 0:
            continue
        condition = ele ** 2
        intersection = intersection + len(np.where(seg * gt == condition)[0])
    seg_num = len(np.where(seg != 0)[0])
    gt_num = len(np.where(gt != 0)[0])

    total_pix = seg_num + gt_num
    Os = seg_num - intersection
    Us = gt_num - intersection

    Dice = (2. * intersection + smooth) / (total_pix + smooth)
    OR = (Os + smooth) / (gt_num + Os + smooth)
    UR = (Us + smooth) / (gt_num + Os + smooth)
    #print((intersection + smooth) / (gt_num + smooth))

    return (Dice,OR,UR)


### 获取标签图片目录数据
def label_data_gene(image_arr, image_path) -> list:
    # image_name_arr = "./data/original_data/lobe/CT"
    image_name_arr = glob.glob(os.path.join(image_path, "*"))
    for index,item in enumerate(image_name_arr):
        if os.path.isdir(item):
            image_arr = label_data_gene(image_arr, item)
        else :
            img = io.imread(item, as_gray=True)
            img = img / 255.0
            img[img > 0.5] = 1
            img[img <= 0.5] = 0
            img = img.flatten()
            image_arr.append(img)
    return image_arr

# read_path = 'E:/Bishe_Code/UnetandUnetPlusPlusandUnetAttention/DataSet_DCM/test/images'    # DICOM 文件
# label_path = 'E:/Bishe_Code/UnetandUnetPlusPlusandUnetAttention/DataSet_DCM/test/labels'  # 灰度图 png文件
# save_path = 'E:/Bishe_Code/UnetandUnetPlusPlusandUnetAttention/DataSet_DCM/result_Unet_DCM_plus5'
test_path = r'D:\CODE\Python_Files\code\code\result_and\Unet_plus2_CTA'    # DICOM 文件
label_path = r'D:\CODE\DataSet\train\vessel\CTA_binary' # 灰度图 png文件

for dir in os.listdir(label_path):
    test_dir = os.path.join(test_path, dir)
    label_dir = os.path.join(label_path, dir)
    print(test_dir, "\n", label_dir)

    # 计算DICE准确率、 过分割率OR、 欠分割率UR
    label_data = []
    result = []
    label_data = label_data_gene(label_data, label_dir)
    result = label_data_gene(result,test_dir)
    dice, OR, UR = getDice(result, label_data)

    # test_file = os.path.join(test_dir + ".txt")
    # fp = open(test_file, "w+")
    # print(">>>> dice: " + str(format(dice, '.4f')), file=fp)
    # print(">>>> OS: " + str(format(OR, '.4f')), file=fp)
    # print(">>>> US: " + str(format(UR, '.4f')), file=fp)

    print(round(dice, 4), '\t', round(OR, 4), '\t', round(UR, 4))
    # fp.close()
