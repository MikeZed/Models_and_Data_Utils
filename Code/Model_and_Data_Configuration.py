

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
IMG_MODE = 'crop'  # options: 'pad', 'patch', 'edges' or 'crop'
IMG_CHANNELS = 1

# ---------------------------------------------------------------------------------------- 

DATA_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows_150"
DATA_FILE =  {'path': r"/home/michael/Cell_Classification/Files/Old_Excel_Files/NormalizedData.xlsx",
			  'all_labels': ["head0", "head1", "head2", "mid0", "head3", "combined"],
              'used_labels': ["head1"], 'data_type': "numeric"} 
              # data_type is mixed, img or numeric  

MODELS_DIR = '/home/michael/Cell_Classification/Model_Objects'

# GRAPHS_DIR = 'Training Graphs'
MODEL_FILE = 'VGG19 binary_crossentropy Transfer, mid0 Adam 123'

# PATH = os.join(MODELS_DIR, MODEL_FILE)


# -------------------------------------------------------------------------------------------------------------------- #
#                                                MODEL PARAMETERS                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

SAVE_MODEL = True    
USE_TRANSFER = True 
# -- training --
TRAIN_VAL_TEST_SPLIT = (80, 20, 0)
USE_GENERATOR = True
BATCH_SIZE = 8
EPOCHS = 300 

# -- transfer learning  --
LAYERS_TO_TRAIN = 5 # options: int / 'all'
USE_TRANSFER = True  


if USE_TRANSFER:
    IMG_CHANNELS = 3

# -- results --
PLOTS_IN_ROW = 2


# -- structure and functions --
# NUM_OF_FILTERS = (32, 64, 256)

LEARNING_RATE = 0.001
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
LOSS_FUNC = 'binary_crossentropy'  # binary_crossentropy, hinge, mean_absolute_error, mean_squared_error 
METRICS = ['binary_accuracy']

if LOSS_FUNC in METRICS:
    METRICS.remove(LOSS_FUNC)


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

]

"""
TRANSFER_MODEL = [


    {'name': 'input', 'input_shape': (867,), 'IO': 'input', 'label': 'in'},

    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.001, 'connected_to': 'in'},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'out1'},
    
    {'name': 'Dense', 'size': 64, 'kernel_regularizer': 0.001, 'connected_to': 'in'},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'out2'},
    
    {'name': 'Concatenate', 'concatenated_layers': ['out1', 'out2']},

    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'}
 ]

"""
    {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN, 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten'},
    {'name': 'DO', 'rate': 0.1},
    
    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 64, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 1},
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

    
