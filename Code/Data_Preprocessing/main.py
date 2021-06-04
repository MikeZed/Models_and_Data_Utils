

"""
                                                     TO-DO                                                             
												  -----------
			* add support for numeric and mixed data in DataSequence:
					* img generator works 
					* check numeric generator 
					* check mixed data generator 
									  
			* take out get_labels part in load_data_generators to another function (add data split as property of Data)

			* change config files to .ini files for multiple configs support 

			* move evaluation to another class 

			* rearrange model evaulation and data code  

			* create dynamic informative model name 


												Possible Additions
											 -----------------------
			* general data handling in DataSequence (multiple images, img name not in first column)

"""
import numpy as np
from PIL import Image
import matplotlib.image
import matplotlib.pyplot as plt
import sklearn
from DataPreprocessor import *
from DataPrepConfig import *
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# ---------------------------------------------------------------------
IMG_SETTINGS = {'img_res': IMAGE_RES, 'img_channels': IMG_CHANNELS,
                'img_preprocessing': IMG_PREPROCESSING}

PREPROCESSING_DATA_SETTINGS = {'img_settings': IMG_SETTINGS, 'data_path': PREPROCESSING_PATH, 'data_file': PREPROCESSING_DATA_FILE,
                               'preprocessing_path': AFTER_PREPROCESSING_PATH, 'rearrange_dataframe_func': rearrange_dataframe}


# ---------------------------------------------------------------------


# np.set_printoptions(threshold=np.inf)


def main():
    '''
    img1=cv2.imread("/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04/D102_F001_C01.png",0)
    print(img1,'\n'*10)
    img1= matplotlib.image.imread("/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04/D102_F001_C01.png")
    print(img1,'\n'*10)
    img1 = plt.imread("/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04/D102_F001_C01.png")
    print(img1,'\n'*10)


    img2=cv2.imread("/home/michael/Cell_Classification/Files/Small_Windows_150/D102_F001_C01.png",0)
    print(img2,'\n'*10)
    img2= matplotlib.image.imread("/home/michael/Cell_Classification/Files/Small_Windows_150/D102_F001_C01.png")
    print(img2,'\n'*10)
    img2 = plt.imread("/home/michael/Cell_Classification/Files/Small_Windows_150/D102_F001_C01.png")
    print(img2,'\n'*10)
    '''

    Data = DataPreprocessor(**PREPROCESSING_DATA_SETTINGS)
    Data.preprocess_data(save_df_to_file=SAVE_DF,
                         to_rearrange_df=REARRANGE_DF, to_save_imgs=SAVE_IMAGES, url=URL)


if __name__ == "__main__":
    main()
