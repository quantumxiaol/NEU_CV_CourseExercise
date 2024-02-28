from mindspore import load_checkpoint, load_param_into_net
import mindspore

from nestedunet import NestedUNet
import numpy as np
import os
import cv2
from mindspore import ops
from mindspore import Tensor

import mindspore

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



def test_data(data):
    # 调用训练的模型
    # model = DeepLabV3(num_classes=2)
    network = NestedUNet()
    
    data_len = data.shape[0]   #CT张数
    result = np.empty((data_len, 512, 512))   #n 512 512


    ckpt = './output_train/unet++_0822/unet++-20_624.ckpt'

    param_dict = load_checkpoint(ckpt)
    # # 将参数加载到网络中
    sigmoid=ops.Sigmoid()
    load_param_into_net(network, param_dict)
    

    for i in range(data_len):   # 0 - n-1 
        

        image_arr = data[i]    # 512 512   某一张CT图片

        image_arr = (image_arr.reshape(1, 1, 512, 512)) / 255.
        image_dataset = Tensor(image_arr, dtype=mindspore.float32)
        # 加载模型
        # # 将模型参数导入parameter的字典中
        # mox.file.copy_parallel(src_url='./output_train', dst_url='s3://southgis-train/output_train')
        #下载训练后的模型
        #关联保存模型的路径
        # mox.file.copy_parallel(src_url='s3://southgis-train/output_train', dst_url='./output_train')
        # 预测的数据
        pre_output = sigmoid(network(image_dataset))        
        pre_arr = pre_output.asnumpy().reshape(1, 512, 512)
        pre_img = pre_arr.reshape((512, 512))
        # print('pre_arr_shape:', pre_arr.shape)
        # print('pre_arr', max(pre_arr[0]))
        # 将ndarry转换为image     
        
        pre_img = np.where(pre_img < 0.5, np.zeros_like(pre_img), np.ones_like(pre_img))     
        
        result[i][:, :] = pre_img
        i += 1 
            
        

    return result


if __name__=="__main__":
    # data = txt2cuboid('CT_arr.txt')
    # DI_cuboid_pic(data, './CT_arr')
#    convert_img_name()
    #print(test_data())
    data = txt2cuboid('CT_arr.txt')   # n 512 512
    # DI_cuboid_pic(data)
    result = test_data(data)   # n 512 512
    print(result.shape)
    print(result[:, 200, 300])
    cuboid2txt(result, 'result_arr.txt')