import numpy as np
#from deeplabv3plus import DeepLab
from nestedunet import NestedUNet
from attention_Unet import AttU_Net
from UNet import UNet

#查看模型参数
def count_params(net):
    """Count  number of parameters in the network
    Args:
        net(mindspre,nn,Cell)
    Returns:
        total_params(int):
    """

    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params


if __name__ == "__main__":
    net2 = NestedUNet()
    net3 = AttU_Net()
    #net4 = UNet(3, 1)
    net4 = UNet()
    params = count_params(net2)
    print(params)
    print(net4)
