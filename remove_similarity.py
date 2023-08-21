import os
import re
import cv2
import numpy as np
import time
import glob, os
from natsort import natsorted

# source = 'C:/GitHub/dds_drone_ws/workspace/training_dds/images'
source = 'C:/DDS_DATA_SET/output/output/notdrone'

print('Start cleaning corrupted images')
images = [f for f in os.listdir(source) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]
images = natsorted( images)

prev_image = ''
for image_file in images:
    if prev_image == '':
        prev_image = image_file
        continue


    current_image_path = source + "/" + image_file
    prev_image_path = source + "/" + prev_image


    img_current = cv2.imread(current_image_path, 0)
    img_prev = cv2.imread(prev_image_path, 0)

    res = cv2.absdiff(img_current, img_prev)

    res = res.astype(np.uint8)

    #--- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100)/ res.size

    is_similar = 'NEW IMAGE'
    if percentage < 70:
        is_similar = 'REMOVE'
        print(prev_image, image_file, percentage, is_similar)
        file_to_remove = source + "/" + image_file
        os.remove(file_to_remove)
        # print(file_to_remove)

    else:
        print(prev_image, image_file, percentage, is_similar)
        prev_image = image_file
    # time.sleep(0.1)
