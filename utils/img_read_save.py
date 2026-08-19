import numpy as np
import cv2
import os
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image, imagename, savepath):
    # 将浮点数转换为 uint8 范围 0-255
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 方法1：如果图像范围在 0-1 之间
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            # 方法2：归一化到 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    imsave(os.path.join(savepath, "{}.png".format(imagename)), image)