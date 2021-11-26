import torch
import torchvision
from torchvision.transforms import transforms as TF
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os
from PIL import Image, ImageDraw
import numpy as np
import re
import json
import albumentations as A
import cv2
from zipfile import ZipFile
from io import BytesIO
import imageio

# json_path = 'D:/data/load_data/test/labels/00110011001.json'
# img_path = 'D:/data/load_data/test/imgs/00110011001.jpg'
# img = cv2.imread(img_path)

img_zip_dir = 'D:/data/load_data/OCR_data/Training/imgs/digital.zip'
ann_zip_dir = 'D:/data/load_data/OCR_data/Training/labels/digital.zip'

class OCRDataset(Dataset):
    def __init__(self, img_zip_dir, ann_zip_dir, img_size=416, is_train=True):
        self.IMG_ZIP_FILE = ZipFile(img_zip_dir, 'r')
        self.ANN_ZIP_FILE = ZipFile(ann_zip_dir, 'r')
        self.img_list = [name for name in self.IMG_ZIP_FILE.namelist() if name.endswith('jpg')]
        self.ann_list = [name for name in self.ANN_ZIP_FILE.namelist() if name.endswith('json')]
        self.img_list.sort()
        self.ann_list.sort()
        self.transform = A.Compose([
                                    A.RandomCrop(width=1000, height=1000),
                                    A.Resize(height=img_size, width=img_size, always_apply=True),
                                    A.RandomBrightnessContrast(p=0.2),
                                    A.Blur(p=0.1),
                                    A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255),
                                    A.ChannelShuffle(p=0.05)
                                ], bbox_params=A.BboxParams(format='yolo',  min_visibility=0.0001, label_fields=['class_labels']))
        self.is_train = is_train

    def float_to_int(self, float_bbox):
        return list(map(lambda x: int(round(x)), float_bbox))

    def voc_to_yolo(self, bbox, height, width):
        xmin, ymin, xmax, ymax = bbox
        bwidth = xmax - xmin
        bheight = ymax - ymin
        xcenter = (bwidth/2) + xmin
        ycenter = (bheight/2) + ymin
        return (xcenter/width, ycenter/height, bwidth/width, bheight/height)

    def __len__(self):
        return len(self.ann_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        ann_path = self.ann_list[idx]
        img = self.IMG_ZIP_FILE.open(img_path)
        img = img.read()
        img = imageio.imread(BytesIO(img))
        h, w, _ = img.shape
        ann = self.ANN_ZIP_FILE.open(ann_path)
        ann = json.load(ann)
        bboxes = ann['text']['word']
        bboxes = list(map(lambda x: self.voc_to_yolo(x['wordbox'], height=h, width=w), bboxes))

        class_labels = ['word']*len(bboxes)

        augmentations = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        image = augmentations['image']
        targets = augmentations['bboxes']
        print(targets)
        #     # for DataLoader
        #     # lables: ndarray -> tensor
        #     # dimension: [batch, cx, cy, w, h, class]
        #     if targets is not None:
        #         targets = torch.zeros((len(labels), 6))
        #         targets[:, 1:] = torch.tensor(labels)
        # else:
        #     targets = labels
        #
        # return image, targets, label_path



ocr_dataset = OCRDataset(img_zip_dir, ann_zip_dir)
ocr_dataset[0]

# with open(json_path, 'r', encoding='utf-8') as json_file:
#     data = json.load(json_file)
# boxes = data['text']['word']
# boxes = list(map(lambda x: x['wordbox'], boxes))
# class_labels = ['word']*len(boxes)
#
#
# transformed = transform(image=img, bboxes=boxes, class_labels=class_labels)
# img_ = transformed['image']
# boxes_ = transformed['bboxes']
#
# def return_uint(tuple_data):
#     return list(map(lambda x: int(round(x)), tuple_data))
# boxes_ = list(map(return_uint, boxes_))
# for box in boxes_:
#     img_ = cv2.rectangle(img_, (box[0], box[1]), (box[2], box[3]), color=(0,0,255), thickness=2)
#
# cv2.imshow('TEST', img_)
# cv2.waitKey()
# cv2.destroyAllWindows()