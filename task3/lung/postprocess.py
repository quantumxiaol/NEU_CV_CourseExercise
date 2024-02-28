from skimage import measure

def lobe_post_processing(image):
    '''
        肺实质后处理
        :param -> image: 肺实质数组
        :return-> array：numpy
    '''
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


