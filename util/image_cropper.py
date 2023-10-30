""" This module is for cropping the face images to remove the background, and only keep the face.  """

import os
import glob

import matplotlib.image as mpimg


def crop_images(src_directory, crop_dimensions=(40, 320, 100, 250)):
    """ This function crops the images in the given directory, and saves them to a new directory. """

    images = glob.glob(os.path.join(src_directory, "*"))
    startx, endx, starty, endy = crop_dimensions
    cropped_images = []

    for image in images:
        image = "./" + image.replace("\\", "/")
        img = mpimg.imread(image, 0)
        img_cropped = img[startx:endx, starty:endy]

        cropped_images.append(img_cropped)

    return cropped_images

