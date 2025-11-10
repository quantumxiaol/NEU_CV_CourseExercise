# 实验3

## 实验目的

熟悉modelarts开发平台，了解Unet++网络原理,了解深度而学习训练及推理流程，掌握肺实质分割方法。

## 实验流程

### 实验3-1

在modelarts上跑通肺实质分割训练及推理代码。
要求:1.训练3个模型即可
上传数据集。
解压数据集。

记得选择正确的目标设备，训练。

保存3个模型。

推理，执行test_copy1.py

### 实验3-2

将推理结果在itk-snap中进行三维重建。

要求:提交三维重建结果截图

如果缺少SimpleITK，pip安装即可。执行pngtonii.py。

### 选做1

将推理结果做后处理操作后在进行三维重建。
提示: 
1.使用最大连通域方法进行后处理
2.使用numpy库函数构建长方体
3.将后处理后的的图像保存
要求:用不同颜色来显示后处理前后的三维重建结果

在pngtonii.py中调用postprocess.py的lobe_post_processing
将allImg输入lobe_post_processing，返回的numpy转为uint8。
