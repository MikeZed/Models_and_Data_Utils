########################################################################################################################
#                                                    TO-DO                                                             #
# TODO add non-generator version support for multiple outputs in functions prepare_data, load_images_and_labels in Data

########################################################################################################################

from Data import Data
from Model import Model
import pandas as pd
import os

from Model_and_Data_Configuration import *

YES = ['y', 'Y', '1', ' ']

MODEL_DICT = {'optimizer': OPTIMIZER, 'loss': LOSS_FUNC, 'metrics': METRICS,
              'struct': MODEL_STRUCT if not USE_TRANSFER else TRANSFER_MODEL, 'layers_to_train': LAYERS_TO_TRAIN}

TRAINING_DICT = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'use_generator': USE_GENERATOR}
# FOLDERS_DICT = {'models': MODELS_DIR, 'graphs': GRAPHS_DIR}

IMG_SETTINGS = {'img_res': IMAGE_RES, 'img_channels': IMG_CHANNELS, 'img_mode': IMG_MODE}


def create_model():
    # checks if there is an existing model,
    # can create a new one by downloading and loading the data, creating a new model, training it and then
    # plots the results and saves them

    # ---------------------------------------
    #            check existing model
    # ---------------------------------------

    # print(MODEL_DICT)

    path = "{}/{}".format(MODELS_DIR, MODEL_FILE)
    path += "" if MODEL_NUM == 0 else "_Num{}".format(MODEL_NUM)

    use_existing = prep_and_check_existing(path)

    # -------------------------------------------------
    #     load existing model or create a new one
    # -------------------------------------------------

    if not use_existing:
        # build and train new model

        if not SAVE_MODEL:
            path = None

        model = Model(img_settings=IMG_SETTINGS)

        load_data_and_construct_model(model=model, url=URL, data_path=DATA_PATH, data_file=DATA_FILE, path=path,
                                      plots_in_row=PLOTS_IN_ROW)

    else:
        # use existing model

        model = Model.load_model(path)

        continue_training = input("Continue training model? (Y/[N]): ")
        continue_training = True if continue_training in YES else False

        if continue_training:
            load_data_and_construct_model(model=model, url=URL, data_path=DATA_PATH, data_file=DATA_FILE, path=path,
                                          plots_in_row=PLOTS_IN_ROW)

        print(model.model.metrics_names)

    return model

# ------------------------------------------------------------------------------------------------------------------- #

def load_data_and_construct_model(model, url, data_path, data_file, path, plots_in_row):
    # load data
    data_loader = Data(use_generator=USE_GENERATOR, split=TRAIN_VAL_TEST_SPLIT, img_settings=IMG_SETTINGS,
                       url=url, data_path=data_path, data_file=data_file, rearrange_dataframe=rearrange_dataframe,
                       preprocessing_path=PREPROCESSING_FOLDER)

    # download data and get dataframe
    data_loader.load_data(save_df_to_file=SAVE_DF, rearrange_df=REARRANGE_DF)

    if SAVE_IMAGES:
        data_loader.save_images()

    # load the data to the memory / create data generators
    data_dict = data_loader.prepare_data(batch_size=TRAINING_DICT['batch_size'])

    # create and train model
    model.construct(**MODEL_DICT, **TRAINING_DICT, data_dict=data_dict, split=TRAIN_VAL_TEST_SPLIT, save_path=path)

    # plot results
    model.plot_results(save_path=path, plots_in_row=plots_in_row)
    
    data_dict, labels = data_loader.prepare_data(get_labels=True, batch_size=TRAINING_DICT['batch_size'])
    
    predictions = model.predict(data_dict)
    
    model.plot_roc(predictions=predictions, labels=labels, save_path=path, plots_in_row=plots_in_row)


def prep_and_check_existing(path):
    # check if there is already an existing model at path
    use_existing = False

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    # if not os.path.exists(GRAPHS_DIR):
    #     os.mkdir(GRAPHS_DIR)

    if os.path.exists(path):
        use_existing = input("A trained model already exists: \nUse existing one? (Y/[N]): ")
        use_existing = True if use_existing in YES else False

        if not use_existing:
            i = 1
            while os.path.exists('{}/{}_Num{}'.format(MODELS_DIR, MODEL_FILE, i)):
                i += 1

            os.rename(path, '{}/{}_Num{}'.format(MODELS_DIR, MODEL_FILE, i))

    return use_existing


def rearrange_dataframe(df): 
    """ 
    readrranges the dataset's dataframe, the output is a dataframe of the following structure:
    
      img_name (feature)    out0 (label)    out1 (label) ...
           img.png              1               2        ...
             .                  .               .
             .                  .               .
             .                  .               .
             
    Note: this function is defined outside of the Data Class because, most likely, different functions would 
          fit different datasets.         
    """
    # --------------------------------------------------------------------------
    pd.options.mode.chained_assignment = None

  #  df = df.loc[~(df['good_pic0'] * df['good_pic1'] == 0)]  # drop bad images
  #  df = df.loc[~(df['head0'] == 'N')]  # drop bad images
  #  df.dropna(inplace=True)

    # df=df[df["head0"] != "N"]

  #  df.drop(['good_pic0', 'good_pic1'], axis=1, inplace=True)

    df.loc[:, 'Identifier'] = df['Identifier'].str.replace('C', 'F')
    df_FC = df['Identifier'].str.split('F', expand=True)

    df.loc[:, 'Identifier'] = [f"D{d}_F{f:03d}_C{c:02d}" for d, f, c in
                               zip(df['Donor'], df_FC[1].astype(int), df_FC[2].astype(int))]

    img_name = df['Identifier'] + ".png"

    df.insert(0, 'img_name', img_name)

    df.drop(['Donor', 'Identifier'], axis=1, inplace=True)

    pd.options.mode.chained_assignment = 'warn'
    # --------------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    return df







