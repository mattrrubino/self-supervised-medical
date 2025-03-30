import numpy as np
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
    


