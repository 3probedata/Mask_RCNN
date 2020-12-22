# -*- coding: utf-8 -*-
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
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

# 本程式為Mask RCNN皮革訓練(Tringing)使用

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
#ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.abspath("../../")

# ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
# 以 mask_rcnn_coco.h5 Weight file為基本網路預訓練權重檔

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#COCO_MODEL_PATH = os.path.join(ROOT_DIR,'logs','shapes20201218T1013',"mask_rcnn_shapes_0100.h5")
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
    NUM_CLASSES = 1 + 9  # background + 1 shapes #如果你有超過一個label, 記得改1 + 2或者以上

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 448

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


config = ShapesConfig()
config.display()


class DrugDataset(utils.Dataset):

    # 取得圖片中多少個物件
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # 解析labelme中得到的yaml文件，從而得到mask每一層對應的Instant Label
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重寫draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # 重新写load_shapes，里面包含自己的自己的类别
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        #Class ['BG','scratch', 'scratch2','Wrinkles', 'Wrinkles2','vein', 'vein2','GrowthPattern','dot','scratch3']
        # Add classes
        self.add_class("shapes", 1, "scratch")
        self.add_class("shapes", 2, "scratch2")
        self.add_class("shapes", 3, "Wrinkles")
        self.add_class("shapes", 4, "Wrinkles2")
        self.add_class("shapes", 5, "vein")
        self.add_class("shapes", 6, "vein3")
        self.add_class("shapes", 7, "GrowthPattern")
        self.add_class("shapes", 8, "dot")
        self.add_class("shapes", 9, "scratch3")

        for i in range(count):
            # 取得圖片寬、高
            print(count, ' :', i)
            filestr = imglist[i].split(".")[0]
            print(filestr, imglist[i])
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = os.path.join(mask_floder, str(filestr) + ".png") # info/cv2_mask/01.png
            yaml_path = os.path.join(dataset_root_path, "labelme_json", filestr + '_json', "info.yaml") # info/labelme_json/01/info.yaml
            print(os.path.join(dataset_root_path, "labelme_json", filestr  + '_json', "img.png")) # info/labelme_json/01/img.png
            cv_img = cv2.imread(os.path.join(dataset_root_path, "labelme_json", filestr + '_json', "img.png")) # info/labelme_json/01/img.png
            #print(os.path.join(dataset_root_path, "labelme_json", filestr, "img.png"))
            cv_img = cv_img[..., ::-1]  # BGR to RGB
            cv_img = cv2.blur(cv_img, (1, 1))
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # 重寫load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
                """
        self.class_names = ['BG','scratch', 'scratch2','Wrinkles', 'Wrinkles2','vein', 'vein2','GrowthPattern','dot','scratch3']
        global iter_num
        # print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("scratch") != -1:
                # print "car"
                labels_form.append("scratch")
            elif labels[i].find("scratch2") != -1:
                # print "car"
                labels_form.append("scratch2")
            elif labels[i].find("Wrinkles") != -1:
                # print "leg"
                labels_form.append("Wrinkles")
            elif labels[i].find("Wrinkles2") != -1:
                # print "leg"
                labels_form.append("Wrinkles2")
            elif labels[i].find("vein") != -1:
                # print "well"
                labels_form.append("vein")
            elif labels[i].find("vein2") != -1:
                # print "well"
                labels_form.append("vein2")
            elif labels[i].find("GrowthPattern") != -1:
                # print "well"
                labels_form.append("GrowthPattern")
            elif labels[i].find("dot") != -1:
                # print "well"
                labels_form.append("dot")
            elif labels[i].find("Wrinkles3") != -1:
                # print "leg"
                labels_form.append("Wrinkles3")
        #print(np.array([self.class_names.index(s) for s in labels_form]))
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        print(class_ids)
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# 基本設定

dataset_root_path = os.path.join(ROOT_DIR, 'samples', 'leather','info') #資料集
img_floder = os.path.join(dataset_root_path, "pic") #圖像
mask_floder = os.path.join(dataset_root_path, "cv2_mask") #MASK
# yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)

# train與val數據集準備
dataset_train = DrugDataset()
dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

# print("dataset_train-->",dataset_train._image_ids)

dataset_val = DrugDataset()
dataset_val.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()

# print("dataset_val-->",dataset_val._image_ids)

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#    image = dataset_train.load_image(image_id)
#    mask, class_ids = dataset_train.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    # print(COCO_MODEL_PATH)
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
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=130,  # 改成你要幾個epoch
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=150,
            layers="all")