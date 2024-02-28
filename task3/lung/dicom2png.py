import os
import SimpleITK
import cv2
import pydicom
import numpy as np
from tqdm import tqdm
from PIL import Image

def is_dicom_file(filename):
    '''
       判断某文件是否是dicom格式的文件
    :param filename: dicom文件的路径
    :return:
    '''
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_patient(src_dir):
    '''
        读取某文件夹内的所有dicom文件
    :param src_dir: dicom文件夹路径
    :return: dicom list
    '''
    files = os.listdir(src_dir)
    slices = []
    for s in files:
        if is_dicom_file(src_dir + '/' + s):
            instance = pydicom.read_file(src_dir + '/' + s)
            slices.append(instance)
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu_by_simpleitk(dicom_dir):
    '''
        读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)
    :param src_dir: dicom文件夹路径
    :return: image array
    '''
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    #img_array[img_array == -2000] = 0
    return img_array

def normalize_hu(image):
    '''
           将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
       :param image 输入的图像数组
       :return: 归一化处理后的图像数组
    '''

    MIN_BOUND = -1000.0
    MAX_BOUND = 600.0

    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
def mark_normalize(image):
    '''
           将输入图像的像素值(-4000 ~ 4000)归一化到0~1之间
       :param image 输入的图像数组
       :return: 归一化处理后的图像数组
    '''
    # image = normalize_hu(image)
    image[image > -1000] = 1.
    image[image <=-1000] = 0.
    return image

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
    if not isExists:
        # 如果不存在则创建目录
     # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        return False


if __name__ == '__main__':
    imagedir = r'D:\CODE\DataSet\Lung\CTA\CTA'
    savedir = r'D:\CODE\DataSet\MindSpore_dataset\train\img\CTA'
    # imagedir = '/Users/fengzihao/desktop/my_work/data/train/dcm/CTA'
    # savedir = '/Users/fengzihao/desktop/my_work/data/train/png/CTA'
    for root, dirlist, filelist in os.walk(imagedir):
        for dir in dirlist:
            dir_path = os.path.join(savedir, dir)
            mkdir(dir_path)
            dicom_dir=os.path.join(root,dir)
            slices = load_patient(dicom_dir)
            print('The number of dicom files : ', len(slices))
            # 提取dicom文件中的像素值
            image = get_pixels_hu_by_simpleitk(dicom_dir)
            for i in tqdm(range(image.shape[0])):

                img_path = os.path.join(dir_path, "1"+str(i).rjust(3, '0') + ".png")
                # 将像素值归一化到[0,1]区间
                org_img = normalize_hu(image[i])

                # 保存图像数组为灰度图(.png)
                cv2.imwrite(img_path, org_img * 255)

                '''
                print(dir_path)
                img_path = os.path.join(dir_path, dir + '-' +"1" + str(i).rjust(3, '0') + ".tif")
                print(img_path)
                # 将像素值归一化到[0,1]区间
                org_img = normalize_hu(image[i])

                # #将dicom保存为0～65535的tiff格式
                im = Image.fromarray((org_img * 65535).astype(np.uint16))
                im.save(img_path)  # --> 16bit(0~65535)
                '''