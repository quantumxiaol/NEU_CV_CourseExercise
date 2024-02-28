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

from rich.progress import DownloadColumn, TransferSpeedColumn, Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn, FileSizeColumn, TotalFileSizeColumn

def mkdir(path):
    # 引入模块
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("/")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    # print(path)
    if not isExists:
        # 如果不存在则创建目录
     # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        return False

def test_data():
    # 网络模型
    # model = DeepLabV3(num_classes=2)
    network = NestedUNet()
    # 测试的图像集路径
    CT_name = 'CT-02'
    test_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR, CT_name)
    # 生成的模型路径
    ckpt_path = os.path.join(cfg.OUTPUT_DIR)
    # 测试图像集与模型
    img_list = os.listdir(test_path)
    ckpt = 'unet++-25_104.ckpt'
    
    print(ckpt_path, os.path.join(ckpt_path, ckpt))

    cnt = 0
    fwiou = 0.
    # 把参数文件中的网络参数加载到网络模型中
    param_dict = load_checkpoint(os.path.join(ckpt_path, ckpt))
    # Sigmoid 函数
    sigmoid=ops.Sigmoid()
    # 把参数字典中相应的参数加载到网络或优化器中
    load_param_into_net(network, param_dict)

    # 进度条
    progress = Progress(TextColumn("[progress.description]{task.description}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    SpinnerColumn(),
                    BarColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn())
    all = progress.add_task(description="进度", total=len(img_list))
    progress.start() ## 开启
    # 逐个图片进行预测
    for image_name in img_list:
        print("当前处理图片", image_name)
        cnt += 1
        # 当前图片位置
        image_path = os.path.join(test_path, image_name)
        # print(image_path)
        # 以灰度模式读取图片（IMREAD_GRAYSCALE = 0，8位深度1通道，形状为：512*512）
        image_arr = cv2.imread(image_path, 0)
        #image_arr = np.transpose(image_arr, (2, 0, 1))
        image_arr = (image_arr.reshape(1, 1, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)) / 255.
        image_dataset = Tensor(image_arr, dtype=mindspore.float32)
        # 加载模型
        # # 将模型参数导入parameter的字典中
        # mox.file.copy_parallel(src_url='./output_train', dst_url='s3://southgis-train/output_train')
        #下载训练后的模型
        #关联保存模型的路径
        # mox.file.copy_parallel(src_url='s3://southgis-train/output_train', dst_url='./output_train')
        # progress.advance(sub, advance=1)
        # 预测的数据
        pre_output = sigmoid(network(image_dataset))       
        # progress.advance(sub, advance=1) 
        pre_arr = pre_output.asnumpy().reshape(1, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)
        pre_img = pre_arr.reshape((cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        # print('pre_arr_shape:', pre_arr.shape)
        # print('pre_arr', max(pre_arr[0]))
        # 将ndarry转换为image        
        pre_img = np.where(pre_img < 0.5, np.zeros_like(pre_img), np.ones_like(pre_img))          
        #label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, image_name.strip('.png')+'_mask.png')
        #label保存的路径
        #gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        #confusion_matrix = generate_matrix(gt_img, pre_img)
        #fwiou += Frequency_Weighted_Intersection_over_Union(confusion_matrix)
        # progress.advance(sub, advance=1)
        #保存图片
        SAVE_DIR = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, 'lung_result', ckpt, CT_name)
        mkdir(SAVE_DIR)
        SAVE_PATH = os.path.join(SAVE_DIR, image_name)
        cv2.imwrite(SAVE_PATH, pre_img * 255)

        progress.advance(all, advance=1)

    return True

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


# 主函数
if __name__=="__main__":

    print(test_data())
