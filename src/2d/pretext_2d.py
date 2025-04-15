import numpy as np
import random
from PIL import Image

# @def rotates a single 2d image and and then returns the classifcation
# the rotation classification for prediction
# @param is a PIL image to be rotated
def rotate_2dimages(image):
    rotation_label = np.random.randint(low =1, high = 5)
    rotation = rotation_label * 90
    image = image.rotate(angle=rotation)

    one_hot = np.zeros((4, 1))
    one_hot[rotation_label -1] = 1

    return image, one_hot
    








def preprocess_image(image, is_training, patches_per_side, patch_jitter, permutations):
    overlap_mode = False
    c, h, w = image.shape

    patch_overlap = 0
    if patch_jitter < 0: # If we are given a negative patch jitter, we actually want overlapping patches
        patch_overlap = -patch_jitter
        overlap_mode = True        

    h_step = (h - patch_overlap)//patches_per_side
    w_step = (w - patch_overlap)//patches_per_side

    patch_height = h_step - patch_jitter
    patch_width = w_step - patch_jitter

    patch_arr = []
    for row in range(patches_per_side):
        for col in range(patches_per_side):
            patch = img_crop(image, row*h_step, col*w_step, h_step+patch_overlap, w_step+patch_overlap)

            if not overlap_mode: # do jitter
                x_patch_start = 0
                y_patch_start = 0
                if is_training:
                    x_patch_start = random.randint(0, patch_jitter)
                    y_patch_start = random.randint(0, patch_jitter)
                else:
                    x_patch_start = patch_jitter // 2
                    y_patch_start = patch_jitter // 2
                patch = img_crop(patch, x_patch_start, y_patch_start, patch_height, patch_width)
            patch_arr.append(patch)

    label = random.randint(0, len(permutations)-1) # permutation label
    y = np.zeros((len(permutations),)) # one hot permutation label
    y[label] = 1

    return np.array(patch_arr)[np.array(permutations[label])], np.array(y)

def img_crop(image, x, y, h, w):
    return image[:, x:(x+h), y:(y+w)]
