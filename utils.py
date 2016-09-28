import numpy as np


def image_grid(imgs):
    """
    Plot all images in a single figure
    :param imgs: nimages x npixels array (assume images are square)
    :return new_image: Images arranged in a square grid (extra images outside square are thrown out)
    """
    nimages = imgs.shape[0]
    img_len = np.sqrt(imgs.shape[1])
    width = int(np.floor(np.sqrt(nimages)))
    height = width
    imgs = [img.reshape((img_len, img_len)) for img in imgs[0: width * height]]
    new_image = np.empty((width*img_len, height*img_len))
    for i in range(0, height):
        for j in range(0, width):
            new_image[i*img_len:(i+1)*img_len, j*img_len:(j+1)*img_len] = imgs[i*width + j]

    return new_image

