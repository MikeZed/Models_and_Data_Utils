

from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd


########################################################################################################################
#                                                   Variables                                                          #
########################################################################################################################

# -------------------------------------------------------------------------------------------------------------------- #
#                                               DATA and FOLDERS                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

URL = None  

SAVE_DF = True
REARRANGE_DF = True
SAVE_IMAGES = True

AFTER_PREPROCESSING_PATH = {'path':  r"/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows"}

PREPROCESSING_PATH = r"/home/michael/Cell_Classification/Files/Large_Windows"

PREPROCESSING_DATA_FILE = r"/home/michael/Cell_Classification/Code/Data_Preprocessing/valid + features (Original).xlsx"
             # 'skip_rows': None, 'relevant_cols': ["img_name", "head0"]}
             #["img_name", "head0",	"head1", "head2", "mid0", "head3"]}
             # relevent_cols = None --> keep all columns
             
# -- image settings --
IMAGE_RES = 150    # setting this parameter for more the 200 is not recommended
IMG_MODE = 'crop'  # options: 'pad', 'patch', 'edges' or 'crop'
IMG_CHANNELS = 1

      
# ---------------------------------------------------------------------------------------- 
# ---------------------------------------------------------------------------------------- 

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
    
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    
    
    

    pd.options.mode.chained_assignment = 'warn'
    # --------------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    return df
    
    
