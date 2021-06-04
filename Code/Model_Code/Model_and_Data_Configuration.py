
import datetime

from tensorflow import keras
import tensorflow as tf

import os

from Model_Structs import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
########################################################################################################################
#                                                   Variables                                                          #
########################################################################################################################

# -------------------------------------------------------------------------------------------------------------------- #
#                                               DATA and FOLDERS                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

DATA_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows_Whitened_23.04/"
# r"/home/michael/Cell_Classification/Files/Small_Windows_Whitened/"
# r"/home/michael/Cell_Classification/Files/Small_Windows_150/"
# r"/home/michael/Cell_Classification/Files/Small_Windows_Whitened_23.04/"

DATA_FILE = {'path': r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx",
             'all_labels': ["head0", "head1", "head2", "mid0", "head3", "combined"],
             'used_labels': ["head0"], 'data_type': "mixed"}
# data_type is mixed, img or numeric

# r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx"
# r"/home/michael/Cell_Classification/Files/valid + features RobustScaler.xlsx"
# r"/home/michael/Cell_Classification/Files/valid + features StandardScaler.xlsx"

MODELS_DIR = '/home/michael/Cell_Classification/models_test'  # Model_Objects'

# None
# "2021-05-16 14:53:52.686132 - VGG19 Small_Windows_Whitened_23.04 ['head0'] mixed binary_crossentropy"
# None
# "2021-05-16 21:23:37.033134 - VGG19 Small_Windows_Whitened_23.04 ['mid0'] mixed binary_crossentropy"
# "123"
# "2021-05-16 14:53:52.686132 - VGG19 Small_Windows_Whitened_23.04 ['head0'] mixed binary_crossentropy"
# "2021-05-19 13:56:25.391359 - VGG19 Small_Windows_Whitened_23.04 ['head0'] mixed binary_crossentropy"
# "2021-05-20 13:31:40.378633 - VGG19 Small_Windows_Whitened_23.04 ['head0'] mixed binary_crossentropy"
# "2021-05-20 13:31:40.378633 - VGG19 Small_Windows_Whitened_23.04 ['head0'] mixed binary_crossentropy"
MODEL_FILE = None
MODEL_NAME_ADDITION = ""
# -------------------------------------------------------------------------------------------------------------------- #
#                                                MODEL PARAMETERS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

SAVE_MODEL = True

# -- training --
TRAIN_VAL_TEST_SPLIT = (80, 20, 0)
BATCH_SIZE = 64
EPOCHS = 10

# -- structure and functions --

LEARNING_RATE = 0.00002  # 0.0005 0.00001
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# binary_crossentropy, hinge, mean_absolute_error, mean_squared_error
LOSS_FUNC = 'binary_crossentropy'
METRICS = ['binary_accuracy', keras.metrics.AUC(
    curve='ROC'), keras.metrics.AUC(curve='PR')]


if LOSS_FUNC in METRICS:
    METRICS.remove(LOSS_FUNC)

MODEL_STRUCT = TRANSFER_MODEL_IMG if (DATA_FILE['data_type'] == 'img') else (
    TRANSFER_MODEL_MIXED if (DATA_FILE['data_type'] == 'mixed') else TRANSFER_MODEL_NUMERIC)

if MODEL_FILE is None:
    MODEL_FILE = "{} - {} {} {} {} {}".format(datetime.datetime.today(),
                                              MODEL_STRUCT[0]["type"],
                                              os.path.basename(
                                                  os.path.normpath(DATA_PATH)),
                                              DATA_FILE["used_labels"],
                                              DATA_FILE["data_type"],
                                              LOSS_FUNC) + MODEL_NAME_ADDITION


callback_path = os.path.join(MODELS_DIR, MODEL_FILE, 'checkpoints')
# 'best_model.hdf5' 'last_50epochs_model.hdf5'
CALLBACKS = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(callback_path, ''),
                                             save_best_only=True,
                                             save_weights_only=False,
                                             monitor='val_auc_1',
                                             mode='max'),

             keras.callbacks.ModelCheckpoint(filepath=os.path.join(callback_path, ''),
                                             save_best_only=False,
                                             save_weights_only=False,
                                             monitor='val_auc_1',
                                             mode='max', save_freq=50*18)]

# -- results --
PLOTS_IN_ROW = 2
