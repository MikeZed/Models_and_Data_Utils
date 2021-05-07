

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

SAVE_DF = False
REARRANGE_DF = False
SAVE_IMAGES = True

AFTER_PREPROCESSING_PATH = r"/home/michael/Cell_Classification/Code/Data_Preprocessing/Small_Windows_Whitened_23.04 123123"

PREPROCESSING_PATH = r"/home/michael/Cell_Classification/Files/Small_Windows_150"

PREPROCESSING_DATA_FILE = r"/home/michael/Cell_Classification/Files/valid + features min-max.xlsx"
             # 'skip_rows': None, 'relevant_cols': ["img_name", "head0"]}
             #["img_name", "head0",	"head1", "head2", "mid0", "head3"]}
             # relevent_cols = None --> keep all columns
             
# -- image settings --
IMAGE_RES = 150    # setting this parameter for more the 200 is not recommended
IMG_PREPROCESSING = ['whiten_imgs']  # options: 'whiten_imgs', 'invert_colors', 'pad' or 'crop' or 'Histogram Equalization'
IMG_CHANNELS = 3

      
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
    df.loc[:, 'Identifier'] = df['Identifier'].str.replace('C', 'F')
    df_FC = df['Identifier'].str.split('F', expand=True)

    df.loc[:, 'Identifier'] = [f"D{d}_F{f:03d}_C{c:02d}" for d, f, c in
                               zip(df['Donor'], df_FC[1].astype(int), df_FC[2].astype(int))]

    img_name = df['Identifier'] + ".png"

    df.insert(0, 'img_name', img_name)

    df.drop(['Donor', 'Identifier'], axis=1, inplace=True)
    
    # x = df.values #returns a numpy array
    

    x = df
    #min max scale

    #min_max_scaler = preprocessing.MinMaxScaler()
    #x.iloc[:,1:-5] = min_max_scaler.fit_transform(x.iloc[:,1:-5].values)
   
    #Standard scaler scale
    #std_scaler = preprocessing.StandardScaler()
    #x.iloc[:,1:-5] = std_scaler.fit_transform(x.iloc[:,1:-5].values)
    
    #RobustScaler
    RobustScaler_scaler = preprocessing.RobustScaler()
    x.iloc[:,1:-5] = RobustScaler_scaler.fit_transform(x.iloc[:,1:-5].values)




    df = pd.DataFrame(x)
    
    
    pd.options.mode.chained_assignment = 'warn'
    # --------------------------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    return df
    
    
