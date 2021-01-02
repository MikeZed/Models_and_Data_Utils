
import warnings

from keras_preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore", category=FutureWarning)

from DataSequence import DataSequence

from image_utils import prepare_image
from tensorflow import keras
import pandas as pd
import numpy as np
import random
import time
import cv2
import os


class Data:
    # used to download the dataset and load it into memory

    def __init__(self, data_path, data_file, use_generator=False, img_settings=None):
    
        self.use_generator = use_generator

        self.data_path = data_path

        self.data_file = data_file
        
        self.img_settings = img_settings
        
        self.df = None


    ####################################################################################################################
    #                                               Loading Data                                                       #
    ####################################################################################################################

    def load_dataframe(self):
        # reads the Ground_Truth file in data_path and creates a pandas dataframe from it
        file_extension = self.data_file['path'].split('.')[-1]

        if file_extension == 'txt' or file_extension == 'csv':
            self.df = pd.read_csv(self.data_file['path'], usecols=self.data_file['relevant_cols'])

        elif file_extension == 'xlsx':
            self.df = pd.read_excel(self.data_file['path'], usecols=self.data_file['relevant_cols'])


    def load_data(self, get_labels=False, batch_size=32, split=(80,10,10)):

        print("Preparing data...")
        
        if self.use_generator:
            data = self.load_data_generators(get_labels, batch_size, split)
        else: 
            data = self.load_data_to_memory(get_labels, batch_size, split)
            
        print("Data is ready!\n") 
        return data

    # ------------------------------------------------------------------------------------------------------------------- #
    def load_data_generators(self, get_labels=False, batch_size=32, split=(80, 10, 10)):
        # prepares the train, val and test data generators for the model

        total_num = len(self.df)

        train_num = int(total_num * split[0] / sum(split))
        train_val_num = int(total_num * (split[0]+split[1]) / sum(split))

        train_df = self.df[0:train_num]
        val_df = self.df[train_num:train_val_num]
        
       # train_df = train_df.sample(frac=1)  # shuffles the training data

        train_df.reset_index(drop=True, inplace=True)
        
       # -----

        y_col = self.df.columns.values.tolist()
        y_col.remove("img_name")
    
        print(y_col)
        class_mode = "multi_output" if len(y_col) > 1 else "raw"
   
        target_size = (self.img_settings["img_res"] ,self.img_settings["img_res"])
        
        color_mode = "grayscale" if self.img_settings["img_channels"] == 1 else "rgb"

        settings = {"directory": self.data_path, "x_col": "img_name", "y_col": y_col, "class_mode": class_mode,
                    "batch_size": batch_size, "target_size": target_size, "color_mode": color_mode}

        data_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, vertical_flip=True)

        train_gen = data_gen.flow_from_dataframe(train_df, **settings)

        val_gen = data_gen.flow_from_dataframe(val_df, **settings)

        """
                settings = {'data_path': self.data_path, 'feature_col': 'IMG', 'label_col': 'SA', 'batch_size': batch_size,
                            'img_settings': self.img_settings}

                train_gen = DataSequence(df=train_df, mode='train', **settings)

                val_gen = DataSequence(df=val_df, mode='val', **settings)

                test_gen = DataSequence(df=test_df, mode='test', **settings)
        """
        # -----

        data_dict = {'train': train_gen, 'val': val_gen}
        
        if train_val_num < total_num: 
            test_df = self.df[train_val_num:]
            test_gen = data_gen.flow_from_dataframe(test_df, **settings)
            data_dict['test'] = test_gen
         
        if get_labels: 
            outputs_num = len(self.df.columns)-1
            
            labels={"train":train_df.drop(columns=["img_name"]).values.tolist(),
                    "val":val_df.drop(columns=["img_name"]).values.tolist()}                         
            print(labels)
            if train_val_num < total_num: 
                labels["test"] = test_df.drop(columns=["img_name"]).values.tolist()  
                
            if outputs_num == 1:
                for key,value in labels.items():
                    labels[key]=list(zip(*value))
                   
            return data_dict, labels
            
        else: 
            return data_dict

    def load_data_to_memory(self, get_labels=False, batch_size=32, split=(80,10,10)):
        # loads all the images(prepared for the model) and skew angles to the memory and
        # splits it to train, val and test according to split

        if self.use_generator:
            return self.prepare_data_generator(get_labels, batch_size)

        start_time = time.time()

        data = self.load_images_and_labels()

       # print(data)

        end_time = time.time()

        print("Time elapsed: {:.2f}s".format(end_time - start_time))

        total_num = len(data)

        train_num = int(total_num * split[0] / sum(split))
        train_val_num = int(total_num * (split[0]+split[1]) / sum(split))

        train_data, val_data = \
            data[0: train_num], data[train_num: train_val_num]

        random.shuffle(train_data)  # shuffle training

        train_img, train_label = zip(*train_data)  # unzip data
        val_img, val_label = zip(*val_data)
        
        train_img = np.array(train_img)
        val_img = np.array(val_img)

        data_dict = {'train': (train_img, train_label), 'val': (val_img, val_label)}
        
        if train_val_num < total_num:
            test_data=data[train_val_num:]
            test_img, test_label = zip(*test_data)  # unzip data
            test_img = np.array(test_img)
            data_dict['test'] = (test_img, test_label)
        
        return data_dict


    # --------------------------------------------------------------------------------------------------------------- #

    def load_images_and_labels(self):
        # loads all the images (prepared for the model) and labels to the memory
        data = [tuple(r) for r in self.df.values.tolist()]
   
        data = (
            (prepare_image("{}/{}".format(self.data_path, img), **self.img_settings), float(label))
            for img, label in data
        )
   
        data = (list(zip(img, [label] * len(img))) for img, label in data)
   
        data = [ex for img_list in data for ex in img_list]
   
        print(data[0][0].shape)
   
        return data
   

            
            
