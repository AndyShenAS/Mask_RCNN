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

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

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

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width, img_floders, mask_floders):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "nuclei")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            self.add_image("shapes", image_id=i, path=img_floders[i],
                           width=width, height=height, mask_path=mask_floders[i], original_info=None)

# >>> a = [{'a':1,'b':2},{'a':1,'b':2}]
# >>> a
# [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}]
# >>> a[0]['a']=100
# >>> a
# [{'a': 100, 'b': 2}, {'a': 1, 'b': 2}]
# >>>


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        # print('smy info : ',info)
        # image = Image.open(img_floders[image_id])  !！!！!！!！!！!！!
        image = Image.open(info['path'])
        image = image.convert("RGB")
        image = np.array(image, dtype=np.uint8)
        # 调整到固定大小
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            mode="square")
        self.image_info[image_id]["original_info"] = (window, scale, padding)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["mask_path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    #重新写draw_mask
    def draw_mask(self, num_obj, image_id, mask, mask_folders):
            info = self.image_info[image_id]
            for index in range(len(mask_folders)):
                image = Image.open(mask_folders[index])
                image = image.convert("RGB")
                image = np.array(image, dtype=np.uint8)
                # 调整到固定大小
                image, window, scale, padding = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode="square")
                image = image[:,:,0]
                image[image>0] = 1
                mask[:, :, index] = image
            return mask

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_path = info['mask_path']
        masklist = os.listdir(mask_path)
        mask_folders = [mask_path+mask for mask in masklist]
        count = len(mask_folders)

        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        mask = self.draw_mask(count, image_id, mask, mask_folders)
        # Handle occlusions  就是把所有mask合成一个mask
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        labels_form=[]
        for i in range(count):
            labels_form.append("nuclei")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask.astype(np.bool), class_ids.astype(np.int32)



# 基础设置
# dataset_root_path="/home/recardo/proj/kaggle/competitions/data-science-bowl-2018/stage1_train/"
# imglist = os.listdir(dataset_root_path)
# imglist_val = []
# for i in range(70):
#     index = random.randint(0,len(imglist)-1)
#     imglist_val.append(imglist[index])
#     imglist.pop(index)
# img_floders = [dataset_root_path+img+"/images/" for img in imglist]
# img_floders = [img+os.listdir(img)[0] for img in img_floders]
# mask_floders = [dataset_root_path+img+"/masks/" for img in imglist]
# img_floders_val = [dataset_root_path+img+"/images/" for img in imglist_val]
# img_floders_val = [img+os.listdir(img)[0] for img in img_floders_val]
# mask_floders_val = [dataset_root_path+img+"/masks/" for img in imglist_val]
# count = len(imglist)
# countval = len(imglist_val)


# print(img_floders_val)
# print(mask_floders_val)
#yaml_floder = dataset_root_path

# print('count: ', count)
# print('countval: ', countval)
# print('img_floders_val.......',len(img_floders_val))
# for i in range(countval):
#     print(img_floders_val[i])
#     print(mask_floders_val[i])

# width = 1024
# height = 1024

# >>> import Image
# >>> im = Image.open("lena.ppm")
# >>> im.show()
# min_dim=config.IMAGE_MIN_DIM,
# max_dim=config.IMAGE_MAX_DIM,
# Training dataset
# dataset_train = ShapesDataset()
# dataset_train.load_shapes(count, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], img_floders, mask_floders)
# dataset_train.prepare()

# Validation dataset
# dataset_val = ShapesDataset()
# dataset_val.load_shapes(countval, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], img_floders_val, mask_floders_val)
# dataset_val.prepare()

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



# image_ids = np.random.choice(dataset_val.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_val.load_image(image_id)
#     mask, class_ids = dataset_val.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names)





# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=15,
#             layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=20,
#             layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
# image_id = random.choice(dataset_val.image_ids)
# # print('ids...',dataset_val.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#     modellib.load_image_gt(dataset_val, inference_config,
#                            image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)

# print(dataset_val.image_info[image_id])


# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_val.class_names, figsize=(8, 8))

# results = model.detect([original_image], verbose=1)
#
# r = results[0]
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
#                             dataset_val.class_names, r['scores'], ax=get_ax())



dataset_root_path="/home/recardo/proj/kaggle/competitions/data-science-bowl-2018/stage2_test_final/"
imglist = os.listdir(dataset_root_path)
img_floders = [dataset_root_path+img+"/images/" for img in imglist]
img_path = [img+os.listdir(img)[0] for img in img_floders]
mask_floders = [dataset_root_path+img+"/masks/" for img in imglist]
count = len(imglist)
for i in range(count):
    isExists=os.path.exists(mask_floders[i])
    if not isExists:
        os.makedirs(mask_floders[i])

dataset_finaltest = ShapesDataset()
dataset_finaltest.load_shapes(count, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], img_path, mask_floders)
dataset_finaltest.prepare()
for i in range(count):
    image_id = i
    test_image = dataset_finaltest.load_image(image_id)
    #smy 显示原图
    # image_show = Image.fromarray(original_image)
    height, width = test_image.shape[:2]
    _, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("")
    ax.imshow(test_image.astype(np.uint8))
    plt.show()
    #smy 显示原图
    results = model.detect([test_image], verbose=1)
    r = results[0]
    visualize.display_instances(test_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_finaltest.class_names, r['scores'], ax=get_ax())
    images = r['masks']
    f = open('./test_parameter.txt','w')
    info = dataset_finaltest.image_info[image_id]
    print(info)

    for j in range(images.shape[2]):
        np.set_printoptions(threshold=np.NaN)
        image = images[:,:,j]
        if image.shape[0] != 1024 or image.shape[1] != 1024:
            break
        print('shape111', image.shape)
        print(info['original_info'])
        image = image[info['original_info'][0][0]:info['original_info'][0][2], info['original_info'][0][1]:info['original_info'][0][3]]
        w = image.shape[0]
        h = image.shape[1]
        print('w&h.....',w,h)
        image = Image.fromarray(255*np.uint8(image))
        print(image.mode)
        image = image.resize((int(w/info['original_info'][1]), int(h/info['original_info'][1])), Image.ANTIALIAS)
        image.save(mask_floders[i]+'{}.jpg'.format(j))
        # if i < 2:
            # image = np.array(image, dtype=np.uint8)
            # image = images[:,:,j]
            # print('shape111', image.shape)
            # print(info['original_info'])
            # image = image[info['original_info'][0][0]:info['original_info'][0][2], info['original_info'][0][1]:info['original_info'][0][3]]
            # f.write(str(image))
            # print('shape...', image.shape)
            # print('haha....')
        # else:
        #     f.close()



        # image, window, scale, padding = utils.resize_image(
        #     image,
        #     min_dim=config.IMAGE_MIN_DIM,
        #     max_dim=config.IMAGE_MAX_DIM,
        #     mode="square")






#
