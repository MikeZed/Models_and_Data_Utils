
# fmt: off 

import sys
sys.path.insert(
    0, '/home/michael/Cell_Classification/Code/Data_Preprocessing/')

from image_utils import rotate_images  # , load_images_to_memory
from tensorflow.keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import random

# fmt: on


# IMG_TYPE=[".png"]
#IMG_KEY_WORD = "img"


class DataSequence(Sequence):
    # used to create data generators to
    # avoid loading all the dataset to the memory
    # although this will result in a much longer training process
    #
    # 'mode'      is 'train', 'val'     or 'test'
    # 'data_type' is 'img',   'numeric' or 'mixed'
    #
    def __init__(self, df, used_labels, all_labels, img_data_path=None, img_settings=None,
                 batch_size=32, mode='train', data_type='img'):

        self.df = df.copy()

        if(mode == 'train'):
            self.df = self.df.sample(frac=1)  # shuffles the training data
            self.df.reset_index(drop=True, inplace=True)

        self.img_data_path = img_data_path
        self.batch_size = batch_size
        self.img_settings = img_settings

        self.mode = mode
        self.data_type = data_type

        feature_cols = [
            col for col in self.df.columns.values if col not in all_labels]
        feature_cols.remove("img_name")

        self.features = self.df[feature_cols].values.tolist()
        self.labels_data = self.df[used_labels].values.tolist()

        if(data_type == 'img' or data_type == 'mixed'):
            self.img_data_gen = self.get_img_data_gen()
        else:
            self.img_data_gen = None

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.labels_data) / self.batch_size))

    def __getitem__(self, idx):
        bsz = self.batch_size

        labels_data = np.array(self.labels_data[bsz * idx: bsz * (idx + 1)])

        if(self.data_type == "mixed"):
            features_data = self.img_data_gen[idx][0]
            features_data = np.array(rotate_images(features_data))

            features_data = [features_data, np.array(
                self.features[bsz * idx: bsz * (idx + 1)])]

        elif(self.data_type == "img"):
            features_data = self.img_data_gen[idx][0]
            features_data = np.array(rotate_images(features_data))

        else:  # self.data_type == "numeric"
            features_data = np.array(self.features[bsz * idx: bsz * (idx + 1)])

        return features_data, labels_data

    def get_img_data_gen(self):
        target_size = (self.img_settings["img_res"],
                       self.img_settings["img_res"])

        color_mode = "grayscale" if self.img_settings["img_channels"] == 1 else "rgb"

        settings = {"directory": self.img_data_path, "x_col": "img_name", "y_col": "img_name", "class_mode": "raw",
                    "batch_size": self.batch_size, "target_size": target_size, "color_mode": color_mode, 'shuffle': False}

        if(self.mode == 'train'):
            data_gen = ImageDataGenerator(
                rescale=1. / 255, vertical_flip=True, horizontal_flip=True)

        else:
            data_gen = ImageDataGenerator(rescale=1. / 255)

        return data_gen.flow_from_dataframe(self.df, **settings)
