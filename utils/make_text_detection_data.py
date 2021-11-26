from zipfile import ZipFile
import json
import os

# test = ZipFile('D:/zipfile_test.zip', 'r')
# filenames = test.namelist()
# for filename in filenames:
#     haha = test.open(filename)
#     print(json.load(haha))

train_labels_save_root = 'D:/data/load_data/text_detection/train/labels'
valid_labels_save_root = 'D:/data/load_data/text_detection/valid/labels'
os.makedirs(train_labels_save_root, exist_ok=True)
os.makedirs(valid_labels_save_root, exist_ok=True)

root = 'D:/data/load_data/OCR_data'
folders = os.listdir(root)
for foldername in folders[:1]:
    label_folder_path = os.path.join(root, foldername, 'labels')
    zipfilenames = os.listdir(label_folder_path)
    for zipfilename in zipfilenames[:1]:
        zipfile_path = os.path.join(label_folder_path, zipfilename)
        zip_root = ZipFile(zipfile_path, 'r')
        filenames = [name for name in zip_root.namelist() if name.endswith('json')]
        for filename in filenames[:1]:
            json_file = zip_root.open(filename)
            data = json.load(json_file)

# zip_root = ZipFile('D:/data/load_data/OCR_data/Training/label_Training_digital.zip', 'r')
# filenames = [name for name in zip_root.namelist() if name.endswith('json')]
#
# for filename in filenames[:1]:
#     subs = filename.split('/')
#     foldername = subs[1]
#     json_filename = subs[2]
#     save_path = os.path.join(tra)
#     json_file = zip_root.open(filename)
#     data = json.load(json_file)
#     width, height = data['image']['width'], data['image']['height']
#     word_boxes = data['text']['word']
#     word_boxes = list(map(lambda x: x['wordbox'], word_boxes))
#     txt_file = open()
#     for word_box in word_boxes: