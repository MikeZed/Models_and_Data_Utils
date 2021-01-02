

from tensorflow import keras
import tensorflow as tf 

########################################################################################################################
#                                                   Variables                                                          #
########################################################################################################################

# -------------------------------------------------------------------------------------------------------------------- #
#                                               DATA and FOLDERS                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

URL = None

SAVE_DF = False
REARRANGE_DF = False
SAVE_IMAGES = False 

PREPROCESSING_PATH = r"/home/michael/Cell_Classification/Files/Large_Windows"

PREPROCESSING_DATA_FILE = {'path':  r"/home/michael/Cell_Classification/Files/data (Corrected) rearranged.xlsx",
             'skip_rows': None, 'relevant_cols': ["img_name", "head1"]}
             #["img_name", "head0",	"head1", "head2", "mid0", "head3"]}
             # relevent_cols = None --> keep all columns
             
# -- image settings --
IMAGE_RES = 150    # setting this parameter for more the 200 is not recommended
IMG_MODE = 'crop'  # options: 'pad', 'patch', 'edges' or 'crop'
IMG_CHANNELS = 1

      
# ---------------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------------- 

DATA_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows_150"
DATA_FILE =  {'path': r"/home/michael/Cell_Classification/Files/data (Corrected) rearranged.xlsx", 'relevant_cols': ["img_name", "head1"]}
MODELS_DIR = '/home/michael/Cell_Classification/Model_Objects'

# GRAPHS_DIR = 'Training Graphs'
MODEL_FILE = 'VGG19 binary_crossentropy 1'

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
EPOCHS = 100


# -- transfer learning  --
LAYERS_TO_TRAIN = 'all' # options: int / 'all'
USE_TRANSFER = False  


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
    
    {'name': 'Dense', 'size': 256},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 128},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 64},
    {'name': 'Activation', 'type': 'relu'},
    {'name': 'BN'},
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'},

]

TRANSFER_MODEL = [

    {'name': 'Transfer', 'type': 'VGG19', 'layers_to_train': LAYERS_TO_TRAIN, 'input_shape': (IMAGE_RES, IMAGE_RES, IMG_CHANNELS), 'IO': 'input'},

    {'name': 'Flatten'},
    {'name': 'DO', 'rate': 0.1},
    
    {'name': 'Dense', 'size': 128},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 64},
    {'name': 'BN'},
    {'name': 'Activation', 'type': 'relu'},
    
    {'name': 'Dense', 'size': 1},
    {'name': 'Activation', 'type': 'sigmoid', 'IO': 'output'},
    

]
"""
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

########################################################################################################################
#                                                   Functions                                                          #
########################################################################################################################


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
    
    
