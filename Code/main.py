

from model_creation import create_model
from image_utils import correct_image_using_model


from Model_and_Data_Configuration import *

MODEL_DICT = {'optimizer': OPTIMIZER, 'loss': LOSS_FUNC, 'metrics': METRICS,
              'struct': MODEL_STRUCT if not USE_TRANSFER else TRANSFER_MODEL, 'layers_to_train': LAYERS_TO_TRAIN}

TRAINING_DICT = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE}

IMG_SETTINGS = {'img_res': IMAGE_RES, 'img_channels': IMG_CHANNELS, 'img_mode': IMG_MODE}

# ------------
MODEL_SETTINGS = {}
DATA_SETTINGS = {}

def main():

    Classifier = create_model(models_dir=MODELS_DIR, model_file=MODEL_FILE, model_num=MODEL_NUM, model_dict=MODEL_DICT,save_model=SAVE_MODEL,use_generator=USE_GENERATOR, img_settings=IMG_SETTINGS,
                 url=URL, data_path=DATA_PATH, data_file=DATA_FILE,preprocessing_path=PREPROCESSING_PATH, training_dict=TRAINING_DICT,  
                 train_val_test_split=TRAIN_VAL_TEST_SPLIT, plots_in_row=PLOTS_IN_ROW,  save_df=SAVE_DF, rearrange_df=REARRANGE_DF,save_images=SAVE_IMAGES, 
                 rearrange_dataframe_func=rearrange_dataframe)

    print("Finishing program...")


if __name__ == "__main__":
    main()
