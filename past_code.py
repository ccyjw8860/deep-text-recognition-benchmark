import codecs
from dataset import OCRDataset
from PIL import Image
import os
import re
import shutil
from tqdm import tqdm
import json
import numpy as np
import os
import sys
import re

import cv2
import six
import math
import lmdb
import torch
import imageio
import json

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, ColorJitter, RandomPerspective, ToPILImage
from io import BytesIO
from zipfile import ZipFile

# sub_dict = {
#     '책표지':'book_cover',
#     '총류':'normal',
#     '철학':'philosophy',
#     '종교':'religion',
#     '사회과학':'social_science',
#     '자연과학':'nature_science',
#     '기술과학':'technology_science',
#     '예술':'art',
#     '언어':'language',
#     '문학':'literature',
#     '역사':'history',
#     '기타':'etc'
# }
#
# img_dir = 'D:/data/OCR_DATASET/val_imgs'
# ann_dir = 'D:/data/OCR_DATASET/val_json'
#
# img_save_dir = 'D:/data/OCR_DATASET/imgs_val'
# ann_save_path = 'D:/data/OCR_DATASET/val_gt.txt'
# os.makedirs(img_save_dir, exist_ok=True)
# img_filenames = os.listdir(img_dir)
#
# gt_file = open(ann_save_path, 'w', encoding='utf-8')
#
# for img_filename in tqdm(img_filenames):
#     ann_filename = re.sub('.jpg', '.json', img_filename)
#     ann_path = os.path.join(ann_dir, ann_filename)
#     filename = ann_filename.split('.')[0]
#     img_path = os.path.join(img_dir, img_filename)
#     if os.path.exists(img_path) and os.path.exists(ann_path):
#         img = Image.open(img_path)
#         w, h = img.size
#         if w>h:
#             with open(ann_path, encoding='utf-8') as json_file:
#                 ann_file = json.load(json_file)
#             anns = ann_file['annotations']
#             i = 0
#             for ann in anns:
#                 text = ann['text']
#                 if text != 'xxx':
#                     bbox = ann['bbox']
#                     new_bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
#                     cropped_img = img.crop(new_bbox)
#                     w_, h_ = cropped_img.size
#                     if w_ >= h_:
#                         w_ = 100
#                         h_ = 32
#                         cropped_img = cropped_img.resize((w_, h_))
#                         cropped_img_name = f'{filename}_{i}.jpg'
#                         i += 1
#                         cropped_img_path = os.path.join(img_save_dir, cropped_img_name)
#                         label = f'imgs_train/{cropped_img_name}\t{text}\n'
#                         cropped_img.save(cropped_img_path)
#                         gt_file.write(label)
#                     else:
#                         pass
#                 else:
#                     pass
#         else:
#             pass
# gt_file.close()

test = OCRDataset(zip_path="D:/data/OCR_DATASET/train.zip")
print(test)
