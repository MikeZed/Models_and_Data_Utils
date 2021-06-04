
# --- img settings ---
IMAGE_RES = 150    # setting this parameter for more the 200 is not recommended
IMG_PREPROCESSING = ['crop']  # options: 'pad' or 'crop'
IMG_CHANNELS = 1

# --- transfer learning ---
USE_TRANSFER = True
LAYERS_TO_TRAIN = 3  # options: int / 'all'

if USE_TRANSFER:
    IMG_CHANNELS = 3

# 'VGG19' 'ResNet50'
# {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN, 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},
# {'name': 'input', 'type': None, 'input_shape': (886,), 'IO': 'input'},
TRANSFER_MODEL_MIXED = [
    {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN,
        'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten', 'label': 'flat'},
    {'name': 'DO', 'rate': 0.1},

    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 32, 'kernel_regularizer': 0.001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'out1'},

    {'name': 'input', 'type': None, 'input_shape': (873,), 'IO': 'input'},

    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.01},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 32, 'kernel_regularizer': 0.01},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'out2'},

    {'name': 'Concatenate', 'concatenated_layers': ['out1', 'out2']},
    {'name': 'DO', 'rate': 0.1},

    {'name': 'Dense', 'size': 16, 'kernel_regularizer': 0.00001},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'}
]

TRANSFER_MODEL_IMG = [
    {'name': 'Transfer', 'type': 'VGG16', 'layers_to_train': LAYERS_TO_TRAIN,
        'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten', 'label': 'flat'},

    {'name': 'Dense', 'size': 1024, 'kernel_regularizer': 0.0005},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 32, 'kernel_regularizer': 0.0005},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu', 'label': 'l1'},

    {'name': 'Dense', 'size': 1, 'connected_to': 'l1'},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'}
]

TRANSFER_MODEL_NUMERIC = [
    {'name': 'input', 'type': None, 'input_shape': (873,), 'IO': 'input'},

    {'name': 'Dense', 'size': 128, 'kernel_regularizer': 0.01},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'Dense', 'size': 16, 'kernel_regularizer': 0.01},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},

    {'name': 'DO', 'rate': 0.1},

    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'}

]

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

MODEL_STRUCT = [

    {'name': 'input', 'input_shape': (
        IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

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
