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

# ct1
# {'unet++-1_500.ckpt': 0.9963015660168483, 
#  'unet++-2_296.ckpt': 0.9969074662698023, 
#  'unet++-3_592.ckpt': 0.9994132139778954, 
#   'unet++-3_92.ckpt': 0.9992721951435363, 
#  'unet++-4_388.ckpt': 0.9994019713172558, 
#  'unet++-5_184.ckpt': 0.9986236878452903}
# {'unet++-5_684.ckpt': 0.9983561914568962, 
#  'unet++-6_480.ckpt': 0.9989475944344082, 
#  'unet++-7_276.ckpt': 0.9986664635039998, 
# ' unet++-8_572.ckpt': 0.9995437795052635, 
#   'unet++-8_72.ckpt': 0.9988118106987286, 
#  'unet++-9_368.ckpt': 0.999516858316463}
#{'unet++-10_164.ckpt': 0.9987537735151876, 
# 'unet++-10_664.ckpt': 0.9991627576874189, 
# 'unet++-11_460.ckpt': 0.9990311494117129, 
# 'unet++-12_256.ckpt': 0.9992244864678105, 
#  'unet++-13_52.ckpt': 0.9984994863328488, 
# 'unet++-13_552.ckpt': 0.9994306031205037}
#{'unet++-14_348.ckpt': 0.9995310158255029,
# 'unet++-15_144.ckpt': 0.9990534249087485, 
# 'unet++-15_644.ckpt': 0.9993564191090213, 
# 'unet++-16_440.ckpt': 0.9993145646279533, 
# 'unet++-17_236.ckpt': 0.9990753118710897, 
#  'unet++-18_32.ckpt': 0.9990087942792296}
#{'unet++-18_532.ckpt': 0.9993266467949355, 
# 'unet++-19_328.ckpt': 0.9994847860524338, 
# 'unet++-20_124.ckpt': 0.9995257132326325, 
# 'unet++-20_624.ckpt': 0.9992255957231824, 
# 'unet++-21_420.ckpt': 0.999346716951091, 
# 'unet++-22_216.ckpt': 0.9991857150827139}
# {'unet++-23_12.ckpt': 0.9990808039941852, 
# 'unet++-23_512.ckpt': 0.9993400765765953, 
# 'unet++-24_308.ckpt': 0.9991736018094651, 
# 'unet++-25_104.ckpt': 0.9996137592612284, 
# 'unet++-25_604.ckpt': 0.999324978670493, 
# 'unet++-26_400.ckpt': 0.9993713868913159}
#{'unet++-27_196.ckpt': 0.9993296976410943, 
# 'unet++-27_696.ckpt': 0.998678401762672, 
# 'unet++-28_492.ckpt': 0.9989173840527085, 
# 'unet++-29_288.ckpt': 0.9992805603340165, 
# 'unet++-30_584.ckpt': 0.9994470676136831, 
#  'unet++-30_84.ckpt': 0.9994398445996773}
#{'unet++-30_704.ckpt': 0.9991467282393754}


# ct2
# {'unet++_15_684.ckpt': 0.9963505897236159, 'unet++_16_480.ckpt': 0.9964975397849773, 'unet++_17_276.ckpt': 0.9963925857396064, 
# 'unet++_18_572.ckpt': 0.9989266672345138, 'unet++_19_72.ckpt': 0.9969518851990216, 'unet++_20_368.ckpt': 0.9979941106494365}

# {'unet++-1_500.ckpt': 0.9782249455097718, 'unet++-2_296.ckpt': 0.9798362929435824, 'unet++-3_592.ckpt': 0.9976240967093098, 
# 'unet++-3_92.ckpt': 0.9970011903900017, 'unet++-4_388.ckpt': 0.9964264075027641, 'unet++-5_184.ckpt': 0.9958302904192384}
# {'unet++-5_684.ckpt': 0.9953724625286737, 'unet++-6_480.ckpt': 0.9973063095094187, 'unet++-7_276.ckpt': 0.9931448940378111, 
# 'unet++-8_572.ckpt': 0.9987753456538496, 'unet++-8_72.ckpt': 0.997458740831788, 'unet++-9_368.ckpt': 0.9977770815016521}
# {'unet++-10_164.ckpt': 0.9966862511396989, 'unet++-10_664.ckpt': 0.9975459765118527, 'unet++-11_460.ckpt': 0.9974603799655435, 
# 'unet++-12_256.ckpt': 0.9971518121511828, 'unet++-13_52.ckpt': 0.9958061630716061, 'unet++-13_552.ckpt': 0.9983926310465006}
# {'unet++-14_348.ckpt': 0.9977351777603298, 'unet++-15_144.ckpt': 0.997486218869368, 'unet++-15_644.ckpt': 0.997966766204643, 
# 'unet++-16_440.ckpt': 0.9981559508313321, 'unet++-17_236.ckpt': 0.9966217542235484, 'unet++-18_32.ckpt': 0.9962196697076094}
# {'unet++-18_532.ckpt': 0.9979972568401705, 'unet++-19_328.ckpt': 0.9973624422679901, 'unet++-20_124.ckpt': 0.9985335449779922, 
# 'unet++-20_624.ckpt': 0.9975969002679254, 'unet++-21_420.ckpt': 0.9970321722604091, 'unet++-22_216.ckpt': 0.9971957734649292}
# {'unet++-23_12.ckpt': 0.9954874507786577, 'unet++-23_512.ckpt': 0.9979633121109756, 'unet++-24_308.ckpt': 0.9952829997053648, 
# 'unet++-25_104.ckpt': 0.9987691843871139, 'unet++-25_604.ckpt': 0.9981623648592642, 'unet++-26_400.ckpt': 0.9962131665091603}
# {'unet++-27_196.ckpt': 0.9976162546406646, 'unet++-27_696.ckpt': 0.9959462229988031, 'unet++-28_492.ckpt': 0.9953273704310939, 
# 'unet++-29_288.ckpt': 0.9964629961173472, 'unet++-30_584.ckpt': 0.9985319550515283, 'unet++-30_84.ckpt': 0.9983149922605148}
# {'unet++-30_704.ckpt': 0.9969059445691062}


# ct3
# {'unet++_15_684.ckpt': 0.9985899906756832, 'unet++_16_480.ckpt': 0.9993423950472047, 'unet++_17_276.ckpt': 0.999768766298557, 
# 'unet++_18_572.ckpt': 0.9996387522896699, 'unet++_19_72.ckpt': 0.9992038968429957, 'unet++_20_368.ckpt': 0.9998023935970187}

# {'unet++-1_500.ckpt': 0.9986930515322152, 'unet++-2_296.ckpt': 0.9995000205104512, 'unet++-3_592.ckpt': 0.9995033546379166, 
# 'unet++-3_92.ckpt': 0.9994702838253424, 'unet++-4_388.ckpt': 0.9996606676158252, 'unet++-5_184.ckpt': 0.9994796769462216}
# {'unet++-5_684.ckpt': 0.9985128373940753, 'unet++-6_480.ckpt': 0.9992597011167011, 'unet++-7_276.ckpt': 0.9997482895951367, 
# 'unet++-8_572.ckpt': 0.9996819084109744, 'unet++-8_72.ckpt': 0.9990477232889596, 'unet++-9_368.ckpt': 0.9998100244770457}
# {'unet++-10_164.ckpt': 0.9996674405632598, 'unet++-10_664.ckpt': 0.9994856207638269, 'unet++-11_460.ckpt': 0.999342754668544, 
# 'unet++-12_256.ckpt': 0.9996678490100446, 'unet++-13_52.ckpt': 0.9991430585611738, 'unet++-13_552.ckpt': 0.9997406279076034}
# {'unet++-14_348.ckpt': 0.9998285221407371, 'unet++-15_144.ckpt': 0.9996703925215736, 'unet++-15_644.ckpt': 0.9995500038234324, 
# 'unet++-16_440.ckpt': 0.9994995929624514, 'unet++-17_236.ckpt': 0.9995479319610177, 'unet++-18_32.ckpt': 0.9995252391815591}
# {'unet++-18_532.ckpt': 0.999448825944038, 'unet++-19_328.ckpt': 0.9998546023757006, 'unet++-20_124.ckpt': 0.9997120163634554, 
# 'unet++-20_624.ckpt': 0.9994214854977238, 'unet++-21_420.ckpt': 0.9996189266895994, 'unet++-22_216.ckpt': 0.9994616090551259}
# {'unet++-23_12.ckpt': 0.9994332169230999, 'unet++-23_512.ckpt': 0.9995147480861108, 'unet++-24_308.ckpt': 0.9998221028246476, 
# 'unet++-25_104.ckpt': 0.9997287286529328, 'unet++-25_604.ckpt': 0.9995160155125044, 'unet++-26_400.ckpt': 0.9997251017425489}
# {'unet++-27_196.ckpt': 0.9994303917715738, 'unet++-27_696.ckpt': 0.9987538082873342, 'unet++-28_492.ckpt': 0.9994383415802348, 
# 'unet++-29_288.ckpt': 0.9998329663002677, 'unet++-30_584.ckpt': 0.9993051358344748, 'unet++-30_84.ckpt': 0.9995654509941415}
# {'unet++-30_704.ckpt': 0.9996417438680829}


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
    

# 测试
def test_data():
    # 调用训练的模型
    # model = DeepLabV3(num_classes=2)
    network = NestedUNet()
    CT_DIR = "CT-01"
    test_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.IMAGE_DIR, CT_DIR)
    ckpt_path = os.path.join(cfg.OUTPUT_DIR)
    # 图像文件列表
    img_list = os.listdir(test_path)
    img_list.sort()
    # 模型文件列表
    ckpt_list = os.listdir(ckpt_path)
    ckpt_list.sort()
    ckpt_list=[ckpt for ckpt in ckpt_list if ckpt.endswith('.ckpt')]
    # 创建空字典
    result = dict()

    # 进度条
    progress = Progress(TextColumn("[progress.description]{task.description}"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                SpinnerColumn(),
                BarColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn())
    all = progress.add_task(description="总进度", total=len(ckpt_list)*len(img_list))
    sub = progress.add_task(description="当前任务", total=len(img_list))
    progress.start() # 开启进度条
    
    # sigmoid函数
    sigmoid=ops.Sigmoid()

    # 对同一组数据测试每一个ckpt
    for ckpt in (ckpt_list):
        print("当前ckpt：", ckpt)
        # 记录图片数
        cnt = 0
        fwiou = 0.
        # 加载当前ckpt
        print("加载参数中...")
        param_dict = load_checkpoint(os.path.join(ckpt_path, ckpt))
        # 将参数加载到网络模型
        print("加载参数到模型中...")
        load_param_into_net(network, param_dict)

        # 逐个图片进行预测
        for image_name in (img_list):
            print("当前处理图片：", image_name)
            cnt += 1
            # 当前图片路径
            image_path = os.path.join(test_path, image_name)
            # print(image_path)
            # 加载彩色图片
            image_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # cv2.imshow('123', image_arr)
            # print(type(image_arr))
            # print("image_arr:", image_arr)
            # image_arr = np.transpose(image_arr, (2, 0, 1))
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

            # 标签路径
            label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, CT_DIR, image_name)
            # label_path = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, cfg.LABEL_DIR, image_name.strip('.png')+'_mask.png')
            # label保存的路径
            gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            confusion_matrix = generate_matrix(gt_img, pre_img)
            fwiou += Frequency_Weighted_Intersection_over_Union(confusion_matrix)

            #保存图片
#             SAVE_DIR = os.path.join(cfg.DATA_DIR, cfg.VALID_DIR, 'lung_result', ckpt, image_name)
#             mkdir(SAVE_DIR)
#             SAVE_PATH = os.path.join(SAVE_DIR, image_name)
#             cv2.imwrite(SAVE_PATH, pre_img * 255)

            progress.advance(sub, advance=1)
            progress.advance(all, advance=1)
        progress.reset(sub)

        result[ckpt] = fwiou / cnt
    progress.advance(sub, advance=len(img_list))
    return result

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
