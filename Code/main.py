########################################################################################################################
#                                                    TO-DO                                                             #
# TODO add non-generator version support for multiple outputs in functions prepare_data, load_images_and_labels in Data

########################################################################################################################

import os
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
