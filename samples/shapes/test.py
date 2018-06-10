import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
# smy
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

config = ShapesConfig()
config.display()

# 基础设置
dataset_root_path="/home/recardo/proj/kaggle/competitions/data-science-bowl-2018/stage1_train/"
imglist = os.listdir(dataset_root_path)
imglist_val = []
# for i in range(70):
#     index = random.randint(0,len(imglist)-1)
#     imglist_val.append(imglist[index])
#     imglist.pop(index)
img_floders = [dataset_root_path+img+"/images/" for img in imglist]
img_floders = [img+os.listdir(img)[0] for img in img_floders]
mask_floders = [dataset_root_path+img+"/masks/" for img in imglist]
img_floders_val = [dataset_root_path+img+"/images/" for img in imglist_val]
mask_floders_val = [dataset_root_path+img+"/masks/" for img in imglist_val]
masklist = os.listdir(mask_floders[0])

for i in range(len(imglist)):
    image = Image.open(img_floders[i])
    image = np.array(image)
    original_shape = image.shape
    if original_shape[0] != original_shape[1]:
        print("smy: ", original_shape,".....",i)

# Load random image and mask.

image = Image.open(img_floders[656])
image = image.convert("RGB")
# mask, class_ids = dataset.load_mask(image_id)
image = np.array(image, dtype=np.uint8)

# mask_path = mask_floders[0]
# masklist = os.listdir(mask_path)
# mask_folders = [mask_path+mask for mask in masklist]
# count = len(mask_folders)
# image = Image.open(mask_folders[0])
# image = image.convert("RGB")
# at_pixel = image.getpixel((100, 100))
# print('at_pixel:',at_pixel)
# image = np.array(image, dtype=np.uint8)


print('image shape:',image.shape)


np.set_printoptions(threshold=np.NaN)
# print(np.nonzero(image))
f = open('./test_parameter.txt','w')
f.write(str(image))
f.close()

# 调整到固定大小
image, window, scale, padding = utils.resize_image(
    image,
    min_dim=config.IMAGE_MIN_DIM,
    max_dim=config.IMAGE_MAX_DIM,
    mode="square")

# image shape: (256, 320, 3)
# image                    shape: (1024, 1024, 3)       min:    0.00000  max:  215.00000  uint8
# window:  (102, 0, 921, 1024)
# scale:  3.2
# padding:  [(102, 103), (0, 0), (0, 0)]



# mask = utils.resize_mask(mask, scale, padding) # mask也要放缩
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)

# Display image and additional stats
# print("image_id: ", image_id, dataset.image_reference(image_id))
log("image", image)
print("window: ", window)
print("scale: ", scale)
print("padding: ", padding)
# log("mask", mask)
# log("class_ids", 1)
# log("bbox", bbox)
# Display image and instances
# visualize.display_instances(image, , , 1, "nuclei")
image=Image.fromarray(image)
image.show()
