
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from image_utils import prepare_image, make_the_images_white
from patoolib import extract_archive
from tensorflow import keras
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image

class DataPreprocessor:
    # used to download the dataset and preprocessing

    def __init__(self, data_path=None, data_file=None, img_settings=None, rearrange_dataframe_func=None, preprocessing_path=None):

        self.data_path = data_path
        self.preprocessing_path = preprocessing_path
        
        self.data_file = data_file

        self.img_settings = img_settings

        self.df = None
        self.rearrange_dataframe_func = rearrange_dataframe_func

    # ---------------------------------------------------------------------------------------------------------------- #

    def preprocess_data(self, save_df_to_file=False, to_rearrange_df=False, to_save_imgs=False,to_whiten_imgs=False, url=None):
        # preprocesses the data: 
        # - downloads the dataset rar and extracts it
        # - loads rearranges the dataframe 
        # - pre-processes the images 
        
        if (not os.path.exists(self.data_path) and url is not None):
            self.download_data(url)

        self.get_dataframe()

        if (to_rearrange_df and self.rearrange_dataframe_func is not None):
            self.df = self.rearrange_dataframe_func(self.df)

        if save_df_to_file:
            path = self.data_file.split('.')[0] + " (1).xlsx"
            self.df.to_excel(path)
        if to_whiten_imgs:
            imgs=make_the_images_white(self.data_path, self.df["img_name"].values.tolist(), self.img_settings["img_channels"])
            self.save_images(imgs=imgs)
        elif to_save_imgs: 
            self.save_images()
            
        print("Data is ready for use.\n")


    def download_data(self, url=None):
        # downloads the data if url is specified

        print("Downloading data...")
        data_path = keras.utils.get_file("DataSet.rar", origin=self.url)  # download rar file
        
        print("Downloading complete!\n"
              "Extracting data...")
        
        extract_archive(data_path)  # extract rar file
        
        print("Extraction complete!\n")


    def get_dataframe(self):
        # loads the dataframe 
        
        file_extension = self.data_file.split('.')[-1]

        if (file_extension == 'txt' or file_extension == 'csv'):
            self.df = pd.read_csv(self.data_file, delim_whitespace=True)

        elif file_extension == 'xlsx':
            self.df = pd.read_excel(self.data_file)


    def save_images(self,imgs=None): 
        # saves all the images (prepared for the model)
        
        os.makedirs(self.preprocessing_path, exist_ok=True) 
        os.makedirs(self.preprocessing_path+'ViewMode', exist_ok=True) 
        
        for i,img_name in enumerate(self.df['img_name']):
            # save image after pre-processing
            
            if imgs is not None:
               #print(imgs[i])
               #break
               save_path = "{}/{}".format(self.preprocessing_path, img_name)
               #cv2.imwrite(save_path, np.array(imgs[i]))
               im = Image.fromarray(imgs[i])
               im.save(save_path)
               
               save_path = "{}/{}".format(self.preprocessing_path+'ViewMode', img_name)
               #cv2.imwrite(save_path, np.array(imgs[i])*255)
               im = Image.fromarray(imgs[i]*255)
               im.save(save_path)
            else:
              img_path = "{}/{}".format(self.data_path, img_name)
              prepare_image(img_path, **self.img_settings, save_img=True, save_path=self.preprocessing_path) 
            
               
            
            
            
            
