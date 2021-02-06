


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
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}



from ModelManager import create_model
from image_utils import correct_image_using_model

from Model_and_Data_Configuration import *

# ---------------------------------------------------------------------
IMG_SETTINGS = {'img_res': IMAGE_RES, 'img_channels': IMG_CHANNELS, 'img_mode': IMG_MODE} 

PREPROCESSING_DATA_SETTINGS = {'img_settings': IMG_SETTINGS, 'url': URL, 'data_path': PREPROCESSING_PATH, 'data_file': PREPROCESSING_DATA_FILE, 
                              'preprocessing_path':DATA_PATH , 'save_df': SAVE_DF, 'rearrange_df': REARRANGE_DF,
                              'save_images': SAVE_IMAGES, 'rearrange_dataframe_func': rearrange_dataframe}
                              

# ---------------------------------------------------------------------

MODEL_DICT = {'optimizer': OPTIMIZER, 'loss': LOSS_FUNC, 'metrics': METRICS,
              'struct': MODEL_STRUCT if not USE_TRANSFER else TRANSFER_MODEL, 'layers_to_train': LAYERS_TO_TRAIN}

TRAINING_DICT = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}



# ------------
MODEL_SETTINGS = {'model_dict': MODEL_DICT,'save_model': SAVE_MODEL}

                                
DATA_SETTINGS = {'training_dict': TRAINING_DICT, 'train_val_test_split': TRAIN_VAL_TEST_SPLIT, 'plots_in_row': PLOTS_IN_ROW,
                'data_path': DATA_PATH, 'data_file': DATA_FILE, 'img_settings': IMG_SETTINGS}
       
       
       
import numpy as np    
import sklearn
import matplotlib.pyplot as plt
    
def main():
 
    Classifier = create_model(MODELS_DIR, MODEL_FILE, **MODEL_SETTINGS, **DATA_SETTINGS)

    print("Finishing program...")


if __name__ == "__main__":
    main()
