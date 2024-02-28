import multiprocessing
import os
import mindspore.dataset as ds
from mindspore import Tensor
import numpy as np
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose
from mydataset import DatasetGenerator
from mindspore import nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import context
from nestedunet import NestedUNet
from FWIoU_metric import EvalCallBack, FWIoU
from loss import WeightedBCELoss
from UNet import UNet
from attention_Unet import AttU_Net
import argparse
# import moxing as mox
import learning_rates
from config import cfg


# 训练代码

def train():

    # 训练图像集
    imgtrain_list = []
    train_dataset_generator = DatasetGenerator(os.path.join(cfg.DATA_DIR, cfg.TRAIN_DIR), cfg.IMAGE_DIR, cfg.LABEL_DIR, imgtrain_list)
    # 测试图像集
    imgtest_list = []
    valid_dataset_generator = DatasetGenerator(os.path.join(cfg.DATA_DIR, cfg.VALID_DIR), cfg.IMAGE_DIR, cfg.LABEL_DIR, imgtest_list)

    train_dataset = ds.GeneratorDataset(train_dataset_generator,["image", "label"], shuffle=False)
    valid_dataset = ds.GeneratorDataset(valid_dataset_generator,["image", "label"], shuffle=False)

    train_dataset = train_dataset.batch(cfg.BATCH_SIZE, num_parallel_workers=1)
    valid_dataset = valid_dataset.batch(cfg.BATCH_SIZE, num_parallel_workers=1)

    # 损失函数
    # loss = WeightedBCELoss(w0=1.39, w1=1.69)
    loss = nn.BCEWithLogitsLoss()
    # loss = nn.DiceLoss()
    loss.add_flags_recursive(fp32=True)

    # 网络模型
    # train_net = UNet()
    train_net = NestedUNet()
    # 不同的网络在这里进行设置，可以选择不同的模型
    # 不同的网络在这里进行设置，可以选择不同的模型
    # 不同的网络在这里进行设置，可以选择不同的模型

    if os.path.exists(cfg.CKPT_PRE_TRAINED):
        param_dict = load_checkpoint(cfg.CKPT_PRE_TRAINED)
        load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = train_dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * cfg.EPOCHS
    if cfg.LR_TYPE == 'cos':
        lr_iter = learning_rates.cosine_lr(cfg.BASE_LR, total_train_steps,
                                           total_train_steps)
    elif cfg.LR_TYPE == 'poly':
        lr_iter = learning_rates.poly_lr(cfg.BASE_LR,
                                         total_train_steps,
                                         total_train_steps,
                                         end_lr=0.0,
                                         power=0.9)
    elif cfg.LR_TYPE == 'exp':
        lr_iter = learning_rates.exponential_lr(cfg.BASE_LR,
                                                cfg.LR_DECAY_STEP,
                                                cfg.LR_DECAY_RATE,
                                                total_train_steps,
                                                staircase=True)
    else:
        raise ValueError('unknown learning rate type')

    opt = nn.SGD(params=train_net.trainable_params(),
                 learning_rate=lr_iter,
                 momentum=0.9,
                 weight_decay=0.0001,
                 loss_scale=cfg.LOSS_SCALE)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(cfg.LOSS_SCALE,
                                               drop_overflow_update=False)
    model = Model(train_net,
                  optimizer=opt,
                  amp_level="O3",
                  loss_fn=loss,
                  loss_scale_manager=manager_loss_scale)
    epoch_per_eval = {"epoch": [], "FWIou": []}
    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()

    # 保存模型
    # save_checkpoint_steps表示每隔多少个step保存一次，keep_checkpoint_max表示最多保留checkpoint文件的数量
    config_ckpt = CheckpointConfig(
        save_checkpoint_steps=cfg.SAVE_CHECKPOINT_STEPS,
        keep_checkpoint_max=cfg.KEEP_CHECKPOINT_MAX)
    # prefix表示生成CheckPoint文件的前缀名；directory：表示存放模型的目录
    cbs_1 = ModelCheckpoint(prefix=cfg.PREFIX,
                            directory=cfg.OUTPUT_DIR,
                            config=config_ckpt)
    # eval_cb = EvalCallBack(model, valid_dataset, cfg.EVAL_PER_EPOCH,
    #                       epoch_per_eval)
    
    # 需要打印的参数列表
    cbs = [time_cb, loss_cb, cbs_1]
    # 训练模型
    model.train(cfg.EPOCHS, train_dataset, callbacks=cbs, dataset_sink_mode=False)

    # mox.file.copy_parallel(src_url=cfg.OUTPUT_DIR,
    #                        dst_url='obs://lqy/img-segment-main/output_train/unet++')

    # mox.file.copy_parallel(src_url=cfg.SUMMARY_DIR,
    #                       dst_url='obs://lqy/img-segment-main/summary_log/unet++')


if __name__ == "__main__":
    # device_id = int(os.getenv('DIVICE_ID', 0))
    # context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False, device_id=device_id)

    # 使用晟腾显卡
    # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    # 使用CPU
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    train()

