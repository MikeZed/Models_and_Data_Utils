

import matplotlib.pyplot as plt
from random import randint
import numpy as np
import cv2
from skimage import exposure
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
#                                           Saving and Loading Images                                                  #
########################################################################################################################


def save_image(image, save_path):
    # im = Image.fromarray((imgs[i] * 255).astype(np.uint8))
    cv2.imwrite(save_path, (image*255).astype(np.uint8))


def load_image(image_path, img_channels=1):
    channels = 0 if img_channels == 1 else 1

    image = cv2.imread(image_path, channels)

    image = np.array(image, dtype="float")

    if img_channels == 1:
        image = np.expand_dims(image, axis=2)

    return image


########################################################################################################################
#                                          Functions that Prepare the Images for the Model                             #
########################################################################################################################

def process_image(image_path, img_res, img_preprocessing, img_channels=1):
    # loads and prepares the image at image_path for the model

    image = load_image(image_path)

    for img_prep in img_preprocessing:

        if img_prep == 'invert_colors':
            image = cv2.bitwise_not(image)

        elif img_prep == 'pad':
            # pad image before resizing it

            padding_size = max(image.shape)

            image = pad_image(image, (padding_size, padding_size), 0)

            image = cv2.resize(image, (img_res, img_res))

            if img_channels == 1:  # if image is grayscale cv2 will remove channel dimension
                image = np.expand_dims(image, axis=2)

        elif img_prep == 'crop':
            # get middle crop of the image, size : img_res X img_res

            shape_x, shape_y = image.shape[0:2]
            crop_region_corner = ((shape_x - img_res) //
                                  2, (shape_y - img_res) // 2)

            image = crop_image(image, win_size=img_res,
                               crop_corner=crop_region_corner)

        elif img_prep == 'contrast streching':
            image = np.array(image, dtype=np.uint8)
            p2, p98 = np.percentile(image, (2, 98))
            image = exposure.rescale_intensity(image, in_range=(p2, p98))

        elif img_prep == 'Histogram Equalization':
            image = np.array(image, dtype=np.uint8)
            image = exposure.equalize_hist(image)

    return image


# -------------------------------------------------------------------------------------------------------------------- #

def whiten_all_images(imgs_path, img_name_list, img_channels):
    # loads all the images to the memory
    # zca_epsilon=1e-6
    # zca_whitening_matrix=None

    data = [np.array(load_image("{}/{}".format(imgs_path, img), img_channels))
            for img in img_name_list]

    mean_mask = np.zeros((150, 150, 3), dtype=float)
    std_mask = np.zeros((150, 150, 3), dtype=float)
    images_num = len(img_name_list)

    for image in data:
        image = image/255
        mean_mask += (image/images_num)

    for image in data:
        image = image/255
        image = abs(image-mean_mask)**2

        std_mask += (image/images_num)

    std_mask = np.sqrt(std_mask)

    images = [((image/255)-mean_mask)/std_mask for image in data]

    images = [((img-img.min())/(img.max()-img.min())) for img in images]

    return images

   # for i,img in enumerate(images):
    # cv2.imshow("ok!",img)  #(img*255)/np.max(img)
    # cv2.waitKey(0)
    # img=((img-img.min())/(img.max()-img.min()))
    # print(i,img.min(),img.max())
    # plt.imshow(img,cmap='gray')
    # plt.show()

    '''
    n = len(data)
    flat_x = np.reshape(data, (n, -1))
    u, s, _ = np.linalg.svd(flat_x.T, full_matrices=False)
    s_inv = np.sqrt(n) / (s + zca_epsilon)
    zca_whitening_matrix = (u * s_inv).dot(u.T)
      
     
    flat_x = data.reshape(-1, np.prod(data.shape[-3:]))
    white_x = flat_x @ zca_whitening_matrix
    data = np.reshape(white_x, data.shape)
    '''


def rotate_images(imgs):
    return [rotate_image(img) for img in imgs]


def rotate_image(img):
    img = ndimage.rotate(img, randint(0, 3) * 90)
    return img


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


def crop_image(image, win_size=90, crop_corner=(0, 0)):
    # crop_corner is the left upper corner of the crop
    crop_img = image[crop_corner[1]: crop_corner[1] +
                     win_size, crop_corner[0]: crop_corner[0] + win_size]
    return crop_img
