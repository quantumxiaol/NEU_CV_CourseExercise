import moxing as mox
from config import cfg

mox.file.copy_parallel(src_url=cfg.OUTPUT_DIR,
                           dst_url='obs://lqy/img-segment-main/output_train/unet++')

mox.file.copy_parallel(src_url=cfg.SUMMARY_DIR,
                          dst_url='obs://lqy/img-segment-main/summary_log/unet++')