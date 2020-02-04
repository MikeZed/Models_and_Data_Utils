

import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

from DataSequence import DataSequence

from image_utils import prepare_image
from patoolib import extract_archive
from tensorflow import keras
import pandas as pd
import numpy as np
import random
import time
import os


class Data:
    # used to download the dataset and load it into memory

    def __init__(self, data_path, data_file, img_settings, url=None, split=(80, 10, 10), use_generator= False, ):
        self.use_generator = use_generator

        self.url = url
        self.data_path = data_path
        self.data_file = data_file
        self.split = split

        self.img_settings = img_settings

        self.df = None

#########################################   Loading Data   #############################################################

    def load_data(self):
        # downloads the dataset rar and extracts it,
        # reads the Ground_Truth file in data_path and creates a pandas dataframe from it

        if not os.path.exists(self.data_path):  # check if folder already exists

            print("Downloading data...")
            data_path = keras.utils.get_file("DataSet.rar", origin=self.url)  # download rar file

            print("Downloading complete!\n"
                  "Extracting data...")
            extract_archive(data_path)  # extract rar file

            print("Extraction complete!\n")

        self.df = pd.read_csv(os.path.join(self.data_path, self.data_file), names=['IMG', '_', 'SA'], skiprows=1,
                              delim_whitespace=True)

        self.df.drop('_', 1, inplace=True)

        #  df['IMG'] = "{}\\{}.tif".format(data_path, df['IMG'])

        self.df['IMG'] += ".tif"

        print(self.df['IMG'])

        print("Data is ready for use.\n")

#########################################   Preparing Data   ###########################################################

    def prepare_data(self, batch_size=32):
        # loads all the images(prepared for the model) and skew angles to the memory and
        # splits it to train, val and test according to split

        print("Preparing data...")

        if self.use_generator:
            return self.prepare_data_generator(batch_size)

        start_time = time.time()

        data = self.load_images()

       # print(data)

        end_time = time.time()

        print("Time elapsed: {:.2f}s".format(end_time - start_time))

        total_num = len(data)

        train_num = int(total_num * self.split[0] / sum(self.split))
        val_num = int(total_num * self.split[1] / sum(self.split))

        train_data, val_data, test_data = \
            data[0: train_num], data[train_num: val_num + train_num], data[val_num + train_num:]

        random.shuffle(train_data)  # shuffle training

        train_img, train_sa = zip(*train_data)  # unzip data
        val_img, val_sa = zip(*val_data)
        test_img, test_sa = zip(*test_data)  # unzip data

        train_img = np.array(train_img)
        val_img = np.array(val_img)
        test_img = np.array(test_img)

        print("Data is ready!")

        data_dict = {'train': (train_img, train_sa), 'val': (val_img, val_sa), 'test': (test_img, test_sa)}

        return data_dict

    def prepare_data_generator(self, batch_size=32):
        # prepares the train, val and test data generators for the model

        total_num = len(self.df)

        train_num = int(total_num * self.split[0] / sum(self.split))
        val_num = int(total_num * self.split[1] / sum(self.split))

        train_df = self.df[0:train_num]
        val_df = self.df[train_num:train_num + val_num]
        test_df = self.df[train_num + val_num:]

        train_df = train_df.sample(frac=1)  # shuffles the training data

        train_df.reset_index(drop=True, inplace=True)

        settings = {'data_path': self.data_path, 'feature_col': 'IMG', 'label_col': 'SA', 'batch_size': batch_size,
                    'img_settings': self.img_settings}

        train_gen = DataSequence(df=train_df, mode='train', **settings)

        val_gen = DataSequence(df=val_df, mode='val', **settings)

        test_gen = DataSequence(df=test_df, mode='test', **settings)

        print("Data is ready!\n")

        data_dict = {'train': train_gen, 'val': val_gen, 'test': test_gen}

        return data_dict

    def load_images(self):
        # loads all the images(prepared for the model) and skew angles to the memory

        data = [tuple(r) for r in self.df.values.tolist()]

        data = (
            (prepare_image("{}\\{}".format(self.data_path, img), **self.img_settings), float(sa))
            for img, sa in data
        )

        data = (list(zip(imgs, [sa] * len(imgs))) for imgs, sa in data)

        data = [ex for img_list in data for ex in img_list]

        print(data[0][0].shape)

        return data

