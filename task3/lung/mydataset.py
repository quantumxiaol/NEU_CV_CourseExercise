import os
import numpy as np
import cv2
from pre_process import data_augment
from config import cfg

'''
# For dataset dir
__C.DATA_DIR = 'MD_DATA/'#划分好的数据集
__C.TRAIN_DIR = 'train'#训练集
__C.VALID_DIR = 'validation'#验证集
__C.IMAGE_DIR = 'images'
__C.LABEL_DIR = 'labels'
__C.RAW_DATA_DIR = '/root/dataset/YG56723/'#官方数据集

# For image
__C.IMAGE_WIDTH = 512#图片大小
__C.IMAGE_HEIGHT = 512
__C.IMAGE_MEAN = [0.304378, 0.364577, 0.315096]#设置默认的mean、std             # BGR
__C.IMAGE_STD = [0.151454, 0.154453, 0.186624]
'''
# 自定义数据集
class DatasetGenerator:
    def __init__(self, root_dir, img_dir, label_dir, img_list):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_dirlist = os.listdir(os.path.join(self.root_dir, self.img_dir))  #CT-01
        self.label_dirlist = os.listdir(os.path.join(self.root_dir, self.label_dir)) #CT-01
        '''
        for root, dirlist, filelist in os.walk(os.path.join(self.root_dir, self.img_dir)):
            for dir in dirlist:
                dir_path = os.path.join(root, dir)
                for x in os.listdir(dir_path):
                    image_list = os.path.join(dir,'-', x)  #CT-01-1000.png
                    img_list.append(image_list)
        '''
        for i in self.img_dirlist:
            image_list_raw = os.listdir(os.path.join(self.root_dir, self.img_dir, i))  #1000.png
            for dir in image_list_raw:
                image = i + '-' + dir  #CT-01-1000.png
                img_list.append(image)
        self.img_list = img_list
        self.label_list = [
            x for x in self.img_list
        ]

    def __getitem__(self, index):
        image_name_raw = self.img_list[index]     #CT-xx-xxxx.png
        image_dir = image_name_raw.split("-")[0] + "-" + image_name_raw.split("-")[1]    #CT-xx
        image_name = image_name_raw.split("-")[2]   #xxxx.png
        image_item_path = os.path.join(self.root_dir, self.img_dir, image_dir, image_name)
        image = cv2.imread(image_item_path, 0)

        label_dir = image_dir
        label_name = image_name
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_dir, label_name)
        label = cv2.imread(label_item_path, 0)

        data_augment(image, label)  #数据增强
        #img = np.transpose(img, (2, 0, 1))
        #label = np.reshape(label, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        image = np.reshape(image, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        label = np.reshape(label, (1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH))
        image = image / 255.
        label = label / 255.
        label[label > 0 ] = 1
        return image, label

    def __len__(self):
        '''
        length = 0
        for i in self.img_dirlist:
            img_list = os.listdir(os.path.join(self.root_dir, self.img_dir, i))
            length += len(img_list)
        return length
        '''
        return len(self.img_list)


if __name__ == '__main__':
    img_list = []
    data = DatasetGenerator(r'D:\CODE\DataSet\MindSpore_dataset\train\CT','img','label', img_list)
    #print(len(data))
    print(data.img_list[10])
    image = data.img_list[10]
    image_dir = image.split("-")[0] + "-" + image.split("-")[1]
    image_name = image.split("-")[2]
    print(image_dir, image_name)
    image_item_path = os.path.join(r'D:\CODE\DataSet\MindSpore_dataset\train\CT', 'img', image_dir, image_name)
    #print(image_item_path)
    image = cv2.imread(image_item_path)
    #cv2.imshow('test', img)
    #cv2.waitKey()
    label_item_path = os.path.join(r'D:\CODE\DataSet\MindSpore_dataset\train\CT', 'label', image_dir, image_name)
    label = cv2.imread(label_item_path)
    image_new, label_new = data_augment(image, label)
    import matplotlib.pyplot as plt
    plt.subplot(221)
    plt.imshow(image)
    plt.subplot(222)
    plt.imshow(label)
    plt.subplot(223)
    plt.imshow(image_new)
    plt.subplot(224)
    plt.imshow(label_new)

    plt.show()