import SimpleITK as sitk
import glob
import numpy as np
from PIL import Image
from postprocess import lobe_post_processing


def save_array_as_nii_volume(data, filename, reference_name=None):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Depth, Height, Width]
        filename: the ouput file name
        reference_name: file name of the reference image of which affine and header are used
    outputs: None
    """
    img = sitk.GetImageFromArray(data)
    if (reference_name is not None):
        img_ref = sitk.ReadImage(reference_name)
        img.CopyInformation(img_ref)
    sitk.WriteImage(img, filename)


def png2nii_main():
    image_path = './result_single'
    image_arr = glob.glob(str(image_path) + str("/*"))
    image_arr.sort()

    print(image_arr, len(image_arr))
    allImg = []
    allImg = np.zeros([541, 512, 512], dtype='uint8')
    for i in range(len(image_arr)):
        single_image_name = image_arr[i]
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)
        allImg[i, :, :] = img_as_np
    # allImg = lobe_post_processing(allImg)
    allImg = np.flip(allImg, axis=1)
    
    # 转uint8
    # allImg = allImg.astype(np.uint8)

    save_array_as_nii_volume(allImg, './result_nii/orig.nii.gz')
    print(np.shape(allImg))
    allImg = np.zeros([541,512, 512], dtype='uint8')


if __name__ == '__main__':
    png2nii_main()
