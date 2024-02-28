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
from utils import *

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
    print(path)
    if not isExists:
        # 如果不存在则创建目录
     # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        return False
    
def get_image_name(str_num):
    len_i = len(str_num)
    if len_i==1:
        name = '100'+str_num
    elif len_i==2:
        name = '10'+str_num
    elif len_i==3:
        name = '1'+str_num
    return name


def convert_img_name(img_dir):
    image_list = os.listdir(img_dir)
    for img in image_list:
        orig_image_name = img.split('.')[0]
        new_image_name = get_image_name(orig_image_name)
        orig_image_path = os.path.join(img_dir, img)
        new_image_path = os.path.join(img_dir, new_image_name+'.png')
        # print(orig_image_path)
        # print(new_image_path)
        os.rename(orig_image_path, new_image_path)

def test_data():
    # 调用训练的模型
    # model = DeepLabV3(num_classes=2)
    network = NestedUNet()
    CT_name = 'CT-01'
    # test_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR, CT_name)
    test_path = './lung_data/test/CT/img/CT-01'
    ckpt_path = os.path.join(cfg.OUTPUT_DIR)
    img_list = os.listdir(test_path)
    # img_list.sort(key=lambda x:int(x.split('.')[0]))
    print(img_list)
    ckpt = 'unet++-1_600.ckpt'
    
    
    #modified 1.17 lpy
    data_num = len(img_list)
    print("data_num:", data_num)
    result = np.empty((data_num, 512, 512))   #n 512 512
    #result = np.empty((200, 512, 512))   #n 512 512
    i = 0
    
    
    print(ckpt_path, os.path.join(ckpt_path, ckpt))
     
    cnt = 0
    fwiou = 0.
    param_dict = load_checkpoint(os.path.join(ckpt_path, ckpt))
    # # 将参数加载到网络中
    sigmoid=ops.Sigmoid()
    load_param_into_net(network, param_dict)
    for image_name in tqdm(img_list):
        cnt += 1
        image_path = os.path.join(test_path, image_name)
        print(image_path)
        image_arr = cv2.imread(image_path, 0)
        print(image_arr.shape)
        #modified by lpy 2023.1.17
        #image_arr = np.transpose(image_arr, (2, 0, 1))
        image_arr = (image_arr.reshape(1, 1, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)) / 255.
        image_dataset = Tensor(image_arr, dtype=mindspore.float32)
        # 加载模型
        # # 将模型参数导入parameter的字典中
        # mox.file.copy_parallel(src_url='./output_train', dst_url='s3://southgis-train/output_train')
        #下载训练后的模型
        #关联保存模型的路径
        # mox.file.copy_parallel(src_url='s3://southgis-train/output_train', dst_url='./output_train')
        # 预测的数据
        pre_output = sigmoid(network(image_dataset))        
        pre_arr = pre_output.asnumpy().reshape(1, cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT)
        pre_img = pre_arr.reshape((cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        # print('pre_arr_shape:', pre_arr.shape)
        # print('pre_arr', max(pre_arr[0]))
        # 将ndarry转换为image
        
        
        
        pre_img = np.where(pre_img < 0.5, np.zeros_like(pre_img), np.ones_like(pre_img))
        pre_img = pre_img.astype(int)
        
        
        
        #label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, image_name.strip('.png')+'_mask.png')
        #label保存的路径
        #gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        #confusion_matrix = generate_matrix(gt_img, pre_img)
        #fwiou += Frequency_Weighted_Intersection_over_Union(confusion_matrix)
        
        result[i][:, :] = pre_img
        i += 1
        
        # modified by lpy1.17
        #保存图片
        # SAVE_DIR = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, 'vessel_result', ckpt, CT_name)
        # mkdir(SAVE_DIR)
        SAVE_DIR = './result_single'
        mkdir(SAVE_DIR)
        SAVE_PATH = os.path.join(SAVE_DIR, image_name)
        print(SAVE_PATH)
        
        cv2.imwrite(SAVE_PATH, pre_img * 255)
        
    result = result.astype(int)    
    return result


if __name__=="__main__":
    # data = txt2cuboid('CT_arr.txt')
    # DI_cuboid_pic(data, './CT_arr')
#    convert_img_name()
    #print(test_data())
    # test_data()
    result =  test_data()

    print(result.shape)
    print(result[:, 200, 300])
    # added by lpy 23.1.17
    print("len(result[:, 200, 300]):", len(result[:, 200, 300]))
    
    #modified by lpy 2023.1.17
    #cuboid2txt(result, 'result_cv2_test_gf.txt')
