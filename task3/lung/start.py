import os
import cv2
import argparse

PWD = os.path.dirname(os.path.abspath(__file__)) #当前的绝对路径


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_name', default="ground1.png", type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--device_target', default="CPU", type=str)
    parser.add_argument('--save_path', default="obs://lqy/MindSpore_dataset/train/CT/img-segment-main/output_train/unet++/", type=str)

    return parser.parse_args()


def local_to_obs(file_path, file):   #本地to obs
    import moxing as mox
    mox.file.copy_parallel(file_path, file)


def start_main():
    print("PWD:", PWD)
    args = get_args_parser()
    print(args.image_name)
    print(args.epochs)
    print(args.batch_size)
    print(args.device_target)
    print(args.save_path)

    image_path = os.path.join(PWD, "pic", args.image_name)

    if os.path.isfile(image_path):
        print("image_path:", image_path)
        f = open("model_test.txt", 'w', encoding='utf-8')
        f.writelines(image_path)
        f.close()

    local_to_obs("model_test.txt", args.save_path)


if __name__ == '__main__':
    start_main()