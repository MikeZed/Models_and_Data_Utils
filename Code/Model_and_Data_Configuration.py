
import datetime

from tensorflow import keras
import tensorflow as tf


########################################################################################################################
#                                                   Variables                                                          #
########################################################################################################################

# -------------------------------------------------------------------------------------------------------------------- #
#                                               DATA and FOLDERS                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

# -- image settings --
IMAGE_RES = 150    # setting this parameter for more the 200 is not recommended
IMG_PREPROCESSING = ['crop']  # options: 'pad' or 'crop'
IMG_CHANNELS = 1

# ---------------------------------------------------------------------------------------- 

DATA_PATH = r"/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04/"
# r"/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened/"
# r"/home/michael/Cell_Classification/Files/Small_Windows_150/"
# r"/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04/"

DATA_FILE =  {'path': r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx",
			  'all_labels': ["head0", "head1", "head2", "mid0", "head3", "combined"],
              'used_labels': ["head1"], 'data_type': "img"}                             
              # data_type is mixed, img or numeric  
              
# r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx"
# r"/home/michael/Cell_Classification/Files/valid + features RobustScaler.xlsx"
# r"/home/michael/Cell_Classification/Files/valid + features StandardScaler.xlsx"

MODELS_DIR = '/home/michael/Cell_Classification/models_test' # Model_Objects'

# GRAPHS_DIR = 'Training Graphs'

# PATH = os.join(MODELS_DIR, MODEL_FILE)


# -------------------------------------------------------------------------------------------------------------------- #
#                                                MODEL PARAMETERS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

SAVE_MODEL = True    
USE_TRANSFER = True 

# -- training --
TRAIN_VAL_TEST_SPLIT = (80, 20, 0)
BATCH_SIZE = 8
EPOCHS = 1

# -- transfer learning  --
LAYERS_TO_TRAIN =  5 # options: int / 'all'
USE_TRANSFER = True  


if USE_TRANSFER:
    IMG_CHANNELS = 3

# -- results --
PLOTS_IN_ROW = 2

# -- structure and functions --

LEARNING_RATE = 0.00001 #   0.0005 0.00001
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNC = 'binary_crossentropy'  # binary_crossentropy, hinge, mean_absolute_error, mean_squared_error 
METRICS = ['binary_accuracy']

if LOSS_FUNC in METRICS:
    METRICS.remove(LOSS_FUNC)
 

# 'VGG19' 'ResNet50'

TRANSFER_MODEL = [

    {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN, 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten', 'label' : 'flat'},
    {'name': 'DO', 'rate': 0.1},


    {'name': 'Dense', 'size': 128,  'connected_to' : 'flat', 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 32, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'}

]

"""
[

    {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN, 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten'},
    {'name': 'DO', 'rate': 0.1},
    
    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 64, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'out2'},

    {'name': 'Dense', 'size': 1, 'connected_to': 'cat'},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'},
    
]
"""
"""
]

    {'name': 'DO', 'rate': 0.3, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.3, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.3, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'DO', 'rate': 0.3, 'connected_to': 1},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid'},

    {'name': 'Outputs', 'outputs': (7, 13, 19, 25, 31)},

]
"""
"""
MODEL_STRUCT = [

    {'name': 'input', 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (5, 5)},
    {'name': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3)},
    {'name': 'MaxPooling2D', 'size': (2, 2)},

    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (3, 3)},
    {'name': 'Conv2D', 'filters': 64, 'kernel_size': (2, 2)},
    {'name': 'MaxPooling2D', 'size': (2, 2)},

    {'name': 'Flatten', 'label': 'flat'},
    {'name': 'DO', 'rate': 0.1},
    
    {'name': 'Dense', 'size': 256},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.001},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 64, 'kernel_regularizer': 0.001},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'},

]
"""
    
MODEL_FILE = "{} - {} {} {} {}".format(datetime.datetime.today(),
                                    TRANSFER_MODEL[0]["type"] if USE_TRANSFER else "non-transfer",
                                    DATA_PATH.split('/')[-1],
                                    DATA_FILE["used_labels"], 
                                    DATA_FILE["data_type"])

