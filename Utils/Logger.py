# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import cv2
import Configuration
from PIL import Image


LOG_DIR_PATH = ''
LOG_FILE = ''


def save_img_cv2(img_name, cv2img):
    # Save image in episode results directory
    if Configuration.PRINT_IMAGES:
        cv2.imwrite(os.path.join(LOG_DIR_PATH, img_name), cv2img)


def save_img(img_name, img_array):
    # Save image in episode results directory
    if Configuration.PRINT_IMAGES:
        im = Image.fromarray(img_array)

        # Save grayscale image
        if len(img_array.shape) < 3:
            im = im.convert("L")

        im.save(os.path.join(LOG_DIR_PATH, img_name))


def write(string):
    # Print string in log file
    LOG_FILE.write("\n" + string)

    # Print string in console
    if Configuration.VERBOSE:
        print(string)
