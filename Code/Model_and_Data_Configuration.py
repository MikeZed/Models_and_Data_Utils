

from tensorflow import keras
import tensorflow as tf 
########################################################################################################################
#                                               DATA and FOLDERS                                                       #
########################################################################################################################
URL = None
DATA_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows"
PREPROCESSING_FOLDER = r"/home/michael/Cell_Classification/Files/Small_Windows_100"

DATA_FILE = {'path':  r"/home/michael/Cell_Classification/Files/data (Corrected) rearranged.xlsx",
             'skip_rows': None, 'relevant_cols': ["img_name", "head0",	"head1", "head2", "mid0", "head3"]}
             # relevent_cols = None --> keep all columns


MODELS_DIR = '/home/michael/Cell_Classification/Model_Objects'

# GRAPHS_DIR = 'Training Graphs'
MODEL_FILE = 'model'
MODEL_NUM = 0

SAVE_DF = False
REARRANGE_DF = False
SAVE_IMAGES = False


########################################################################################################################
#                                                MODEL PARAMETERS                                                      #
########################################################################################################################

SAVE_MODEL = True  

# -- training --
TRAIN_VAL_TEST_SPLIT = (80, 10, 10)
USE_GENERATOR = True
BATCH_SIZE = 8
EPOCHS = 150


# -- transfer learning  --
LAYERS_TO_TRAIN = 30
USE_TRANSFER = True


# -- image settings --
IMAGE_RES = 100  # setting this parameter for more the 200 is not recommended
IMG_MODE = 'crop'  # 'pad', 'patch', 'edges' or 'crop'
IMG_CHANNELS = 1

if USE_TRANSFER:
    IMG_CHANNELS = 3

# -- results --
PLOTS_IN_ROW = 3


# -- structure and functions --
# NUM_OF_FILTERS = (32, 64, 256)

LEARNING_RATE = 0.0001
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNC = 'binary_crossentropy'  # binary_crossentropy, hinge
METRICS = ['binary_accuracy']

if LOSS_FUNC in METRICS:
    METRICS.remove(LOSS_FUNC)


MODEL_STRUCT = [

    {'name': 'input'},

    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (5, 5)},
    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3)},
    {'name': 'MaxPooling2D', 'size': (2, 2)},

    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3)},
    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (2, 2)},
    {'name': 'MaxPooling2D', 'size': (2, 2)},

    {'name': 'Flatten'},
    {'name': 'DO', 'rate': 0.1, 'connected_to': 7},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 7},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 7},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 7},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 7},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'Outputs', 'outputs': (12, 17, 22, 27, 32)},
]

TRANSFER_MODEL = [

    {'name': 'ResNet50'},

    {'name': 'Flatten'},
    {'name': 'DO', 'rate': 0.1, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.1, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'Outputs', 'outputs': (7, 13, 19, 25, 31)},

]

