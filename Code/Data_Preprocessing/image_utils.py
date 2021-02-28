

import matplotlib.pyplot as plt
from random import randint
import numpy as np
import cv2


########################################################################################################################
#                                          Functions that Prepare the Images for the Model                             #
########################################################################################################################

def prepare_image(image_path, img_res, img_mode, img_channels=1, vertical_flip=False, horizontal_flip=False, invert_colors=False, save_img=False, save_path=None):
    # loads and prepares the image at image_path for the model

    channels = 0 if img_channels == 1 else 1

    image = cv2.imread(image_path, channels)

    if invert_colors:
        image = cv2.bitwise_not(image)

    if not save_img:
        image = image / 255.0

    image = np.array(image, dtype="float")

    if img_channels == 1:
        image = np.expand_dims(image, axis=2)

    if img_mode == 'pad':
        # pad image before resizing it

        padding_size = max(image.shape)

        image = pad_image(image, (padding_size, padding_size), 0)

        image = cv2.resize(image, (img_res, img_res))

        if img_channels == 1:  # if image is grayscale cv2 will remove channel dimension
            image = np.expand_dims(image, axis=2)

        images = [image]

    elif img_mode == 'patch':
        # get patches out of the image

        shape = (img_res, img_res)
        images = get_patches(image, shape)

    elif img_mode == 'edges':
        # get a patch in the left side of the image where the is the edge of the image

        shape = (img_res, img_res)
        images = get_edges(image, shape)
    else:
        images = [image]

    if img_mode == 'crop':
        # get middle crop o the image, size : img_res X img_res

        shape_x, shape_y = image.shape[0:2]
        crop_region_corner = ((shape_x - img_res) // 2, (shape_y - img_res) // 2)

        image = crop_image(image, win_size=img_res, crop_corner=crop_region_corner)
        images = [image]

    if save_img:
        # get img name and type
        img_name = image_path.split('/')[-1]
        img_name, img_type = img_name.split('.')

        # get image save path
        if len(images) == 1:
            save_path = ["{}/{}.{}".format(save_path, img_name, img_type)]
        else:
            save_path = ["{}/{}{}.{}".format(save_path, img_name, i, img_type) for i in range(len(images))]

        # save image
        for i in range(len(images)):
            cv2.imwrite(save_path[i], images[i])

    return images


# -------------------------------------------------------------------------------------------------------------------- #

def pad_image(image, size, val):
    # pads image to a square shape
    h, w = size
    oh, ow, _ = image.shape

    if not size >= image.shape:
        raise ValueError("Padding height or width are too small.")

    rows_dim_add = (h - oh) // 2
    cols_dim_add = (w - ow) // 2

    image = cv2.copyMakeBorder(image, top=rows_dim_add, bottom=rows_dim_add, right=cols_dim_add, left=cols_dim_add,
                               borderType=cv2.BORDER_CONSTANT, value=val)

    return image


def get_patches(image, shape=(100,100), patch_amount=16, num_of_trials=100, con_range=(0.2,0.8)): 
    # gets patches from the image

    patches = []

    h, w, _ = image.shape

    for i in range(num_of_trials):

        if len(patches) >= patch_amount:
            break

        row_offset = randint(0, h - (1 + shape[0]))
        col_offset = randint(0, w - (1 + shape[1]))

        patch = np.copy(image[row_offset: row_offset+shape[0], col_offset: col_offset+shape[1]])

        patch_con = np.mean(patch)

        if con_range[0] < patch_con < con_range[1]:
            patches.append(patch)

    return patches


def get_edges(image, shape=(100,100), tran_len = 1, depth=1000, con_range=(0.015,1)):
    # tries to get left edge of the image

    h, w = image.shape

    row_offset =(h-shape[0])//2
    col_offset = 0
    translate = 0

    left = np.copy(image[row_offset: row_offset+shape[0], col_offset: col_offset+shape[1]])

    for d in range(depth):

        col_offset += translate

        p_left = np.copy(image[row_offset: row_offset+shape[0], col_offset: col_offset+shape[1]])

        con = np.mean(p_left)

        if con_range[0] < con < con_range[1] and np.sum(left,axis=0)[shape[1]//2]>1 and p_left.shape == shape:
            left=p_left
            break

        translate = d*tran_len

    return [left]


def crop_image(image, win_size = 90, crop_corner = (0,0)):
    # crop_corner is the left upper corner of the crop
    crop_img = image[crop_corner[1]: crop_corner[1] + win_size, crop_corner[0]: crop_corner[0] + win_size]
    return crop_img


########################################################################################################################
#                                         Using Model for Correcting Images                                            #
########################################################################################################################

def correct_image_using_model(model, image_path):
    # Receives a path to an image and then estimates the skew of the image according to the model,
    # corrects the image and plots the original and corrected image, returns corrected image

    image = prepare_image(image_path, **model.img_settings)
    print (image)
    skew_angle = (np.mean(model.model.predict(np.array(image))))
    print(skew_angle)
    print("According to the model the skew of the image at {} is: {:.3f}".format(image_path, skew_angle))

    image = cv2.imread(image_path, 0)

    rows, cols = image.shape

    image = cv2.bitwise_not(image)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -1 * skew_angle, 1)  # finds rotation matrix and
    correct_image = cv2.warpAffine(image, M, (cols, rows))  # rotates the image

    image = cv2.bitwise_not(image)
    correct_image = cv2.bitwise_not(correct_image)

    plt.gray()

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(correct_image)
    plt.title('Corrected Image')

    plt.show()

    return correct_image
