from mindspore import load_checkpoint, load_param_into_net
import mindspore
from UNet import UNet
from attention_Unet import AttU_Net
from nestedunet import NestedUNet
import numpy as np
import os
import cv2
from mindspore import ops
from mindspore import Tensor
from config import cfg
import mindspore
from tqdm import tqdm

# 精度测试
def testvalue():
    # 图像文件列表
    test_dir = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR, 'CT-01')
    pre_dir = "E:\DUCHUANG\lung_data\test\CT\lung_result\unet++-10_664.ckpt\CT-01"
    # os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, 'lung_result', cfg.IMAGE_DIR, 'CT-01')
    cnt = 0
    fwiou = 0.
    for image_name in tqdm(test_path):
        print("当前处理文件：", image_name)
        cnt += 1
        test_path = os.path.join(test_dir, image_name)
        pre_path = os.path.join(pre_dir, image_name)

        test_img = cv2.imread(test_path, cv2.IMREAD_COLOR)
        pre_img = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)


        test_img = np.transpose(test_img, (2, 0, 1))
        test_img = (test_img.reshape(1, 3, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)) / 255.
        pre_img = np.where(pre_img < 0.5, np.zeros_like(pre_img), np.ones_like(pre_img))




        label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, image_name.strip('.png')+'_mask.png')
        #label保存的路径
        gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        confusion_matrix = generate_matrix(gt_img, pre_img)
        fwiou += Frequency_Weighted_Intersection_over_Union(confusion_matrix)
    return fwiou / cnt

def generate_matrix(gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0.) & (gt_image < num_class)

        lab = num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(lab, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class,
                                         num_class)  # 2 * 2(for pascal)
        # print(confusion_matrix)
        return confusion_matrix

# FWIoU计算
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) +
                                        np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

if __name__=="__main__":
    print(test_data())
