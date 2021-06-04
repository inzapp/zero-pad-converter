import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


target_size = (608, 416)

paths = glob('*.jpg')
all_label_exists = True
for path in paths:
    label_path = f'{path[:-4]}.txt'
    if not (os.path.exists(label_path) and os.path.isfile(label_path)):
        print(f'label not exists : {label_path}')
        all_label_exists = False
if not all_label_exists:
    exit(0)

target_ratio = target_size[0] / float(target_size[1])
for path in tqdm(paths):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    raw_height, raw_width = img.shape[0], img.shape[1]

    pad = 0
    lr_pad = False
    raw_ratio = raw_width / float(raw_height)
    width_before_resize = 0
    height_before_resize = 0
    if raw_ratio == target_ratio:
        continue
    elif raw_ratio > target_ratio:
        lr_pad = False
        height_before_resize = int(raw_width / target_ratio)
        pad = int((height_before_resize - raw_height) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif raw_ratio < target_ratio:
        lr_pad = True
        width_before_resize = int(raw_height * target_ratio)
        pad = int((width_before_resize - raw_width) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))
    if raw_width > target_size[0] or raw_height > target_size[1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    label_path = f'{path[:-4]}.txt'
    with open(label_path, 'rt') as f:
        lines = f.readlines()

    s = ''
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.split()))
        class_index = int(class_index)

        pad_start_ratio = 0.0
        if lr_pad:
            pad_start_ratio = pad / float(width_before_resize)
        else:
            pad_start_ratio = pad / float(height_before_resize)
        pad_end_ratio = 1.0 - pad_start_ratio
        weight = pad_end_ratio - pad_start_ratio
        if lr_pad:
            cx = cx * weight + pad_start_ratio
            w *= weight
        else:
            cy = cy * weight + pad_start_ratio
            h *= weight

        cx, cy, w, h = np.clip(np.array([cx, cy, w, h]), 0.0, 1.0)
        s += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'

    cv2.imwrite(path, img)
    with open(label_path, 'wt') as f:
        f.writelines(s)
