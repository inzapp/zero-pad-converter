import numpy as np

from cv2 import cv2
from glob import glob
from tqdm import tqdm


paths = glob('*.jpg')

for path in tqdm(paths):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    raw_height, raw_width = img.shape[0], img.shape[1]

    pad = 0
    lr_pad = False
    square_wh = 0
    if raw_height == raw_width:
        continue
    elif raw_height > raw_width:
        square_wh = raw_height
        pad = int((raw_height - raw_width) / 2)
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))
        lr_pad = True
    elif raw_height < raw_width:
        square_wh = raw_width
        pad = int((raw_width - raw_height) / 2)
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

    label_path = f'{path[:-4]}.txt'
    with open(label_path, 'rt') as f:
        lines = f.readlines()

    s = ''
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.split()))
        class_index = int(class_index)

        pad_start_ratio = pad / float(square_wh)
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

