
import warnings

from keras_preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore", category=FutureWarning)

from DataSequence import DataSequence

from tensorflow import keras
import pandas as pd
import numpy as np
import random
import time
import cv2
import os


class Data:
    # used to create the data generators 

    def __init__(self, data_path, data_file, img_settings=None):
    
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
            self.df = pd.read_csv(self.data_file['path'])

        elif file_extension == 'xlsx':
            self.df = pd.read_excel(self.data_file['path'])


    def load_data(self, batch_size=32, split=(80,10,10), mode='train'):

        print("Preparing data...")
        
        data = self.load_data_generators(batch_size, split, mode)

        print("Data is ready!\n") 

        return data

    # ------------------------------------------------------------------------------------------------------------------- #

    def load_data_generators(self, batch_size=32, split=(80, 10, 10), mode='train'):
        # prepares the train, val and test data generators for the model
        train_df, val_df, test_df = self.get_dataframe_split(split) 

        settings = {'img_data_path': self.data_path, 'used_labels': self.data_file['used_labels'], 'all_labels': self.data_file['all_labels'],
                    'data_type': self.data_file['data_type'], 'batch_size': batch_size, 'img_settings': self.img_settings}

        train_gen = DataSequence(df=train_df, mode=mode, **settings)

        val_gen = DataSequence(df=val_df, mode='val', **settings)

        data_dict = {'train': train_gen, 'val': val_gen}
        
        if len(test_df) > 0: 
            data_dict['test'] = DataSequence(df=test_df, mode='test', **settings)

        return data_dict

    def get_labels(self, split):
        train_df, val_df, test_df = self.get_dataframe_split(split) 

        outputs_num = len(self.data_file['used_labels']) 
            
        labels={"train": train_df[self.data_file['used_labels']].values.tolist(),
                "val":val_df[self.data_file['used_labels']].values.tolist()}                         

        if len(test_df) > 0: 
            labels["test"] = test_df.drop(columns=["img_name"]).values.tolist()  
                
        if outputs_num == 1:
            for key,value in labels.items():
                labels[key]=list(zip(*value))

        return labels

    def get_dataframe_split(self, split):
        total_num = len(self.df)

        train_num = int(total_num * split[0] / sum(split))
        train_val_num = int(total_num * (split[0]+split[1]) / sum(split))

        train_df = self.df[0:train_num]
        val_df = self.df[train_num:train_val_num]
        test_df = self.df[train_val_num:]

        return train_df, val_df, test_df

        