
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from image_utils import prepare_image
from patoolib import extract_archive
from tensorflow import keras
import pandas as pd
import os


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

    def preprocess_data(self, save_df_to_file=False, to_rearrange_df=False, to_save_imgs=False, url=None):
        # preprocesses the data: 
        # - downloads the dataset rar and extracts it
        # - loads rearranges the dataframe 
        # - pre-processes the images 
        
        if (not os.path.exists(self.data_path) and url is None):
            self.download_data(url)

        self.get_dataframe()

        if (to_rearrange_df and self.rearrange_dataframe_func is not None):
            self.df = self.rearrange_dataframe_func(self.df)

        if save_df_to_file:
            path = self.data_file['path'].split('.')[0] + " (1).xlsx"
            self.df.to_excel(path)
            
        if to_save_imgs: 
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
        
        file_extension = self.data_file['path'].split('.')[-1]

        if (file_extension == 'txt' or file_extension == 'csv'):
            self.df = pd.read_csv(self.data_file, usecols=self.data_file['relevant_cols'],
                                  skiprows=self.data_file['skip_rows'], delim_whitespace=True)

        elif file_extension == 'xlsx':
            self.df = pd.read_excel(self.data_file["path"], usecols=self.data_file['relevant_cols'],
                                    skiprows=self.data_file['skip_rows'])


    def save_images(self): 
        # saves all the images (prepared for the model)
        
        os.makedirs(self.preprocessing_path, exist_ok=True) 
        
        for img_name in self.df['img_name']:
            # save image after pre-processing
            img_path = "{}/{}".format(self.data_path, img_name)
            prepare_image(img_path, **self.img_settings, save_img=True, save_path=self.preprocessing_path) 
            
            
            
            
