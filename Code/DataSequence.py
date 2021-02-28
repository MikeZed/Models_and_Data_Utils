

from tensorflow.keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import random


#IMG_TYPE=[".png"]
#IMG_KEY_WORD = "img"

class DataSequence(Sequence):
    # used to create data generators to
    # avoid loading all the dataset to the memory
    # although this will result in a much longer training process
    #
    # 'mode'      is 'train', 'val'     or 'test'
    # 'data_type' is 'img',   'numeric' or 'mixed'
    # 
    def __init__(self, df, used_labels, all_labels, img_data_path = None, img_settings = None, 
                 batch_size=32, mode='train', data_type = 'img'):

        self.df = df.copy()

        self.img_data_path = img_data_path
        self.batch_size = batch_size
        self.img_settings = img_settings
        
        self.mode = mode
        self.data_type = data_type 

        feature_cols = [col for col in df.columns.values if col not in all_labels]
        feature_cols.remove("img_name")

        #self.df_img_names = df["img_name"]
        self.features = df[feature_cols].values.tolist()
        self.labels_data = df[used_labels].values.tolist()

        
        if(mode == 'train'):
            self.df = self.df.sample(frac=1)  # shuffles the training data
            self.df.reset_index(drop=True, inplace=True)
        

        if(data_type == 'img' or data_type == 'mixed'): 
            self.img_data_gen = self.get_img_data_gen()
        else: 
            self.img_data_gen = None 
       
        """
        # self.f_type=["img" if IMG_KEY_WORD in f else "float" for f in df[feature_cols].columns]
        features = df[feature_cols].to_list()
       


        
        self.data = (features, labels)
        """

    def on_epoch_end(self):
    	pass 
    	"""
        if not self.mode == 'train':
            return
        
        self.data=list(zip(self.data[0], self.data[1]))
        random.shuffle(self.data)                        # shuffle
        self.data=list(zip(*self.data))
		"""

    def __len__(self):
        return int(np.ceil(len(self.labels_data) / self.batch_size))

    def __getitem__(self, idx):
        bsz = self.batch_size

        labels_data = self.labels_data[bsz * idx: bsz * (idx + 1)]

        if(self.data_type == "mixed"):
            features_data = self.img_data_gen[idx][0] 
            features_data = list(zip(features_data, self.features[bsz * idx: bsz * (idx + 1)]))

        elif(self.data_type == "img"): 
            features_data = self.img_data_gen[idx][0]  

        else: # self.data_type == "numeric" 
            features_data = self.features[bsz * idx: bsz * (idx + 1)]

        # data_batch = self.data[bsz * idx: bsz * (idx + 1)]
        #labels = self.data[1][bsz * idx: bsz * (idx + 1)]
        """
        f = features[:]
        load_img=False 
        
        for j in range(len(features[0])): 
            if self.f_type[j] == "img":
                load_img=True 
            for i in range(len(features)): 
                if load_img: 
                  f[i][j]=prepare_image("{}/{}".format(self.data_paths[j],img), **self.img_settings)
        """
        """
        data_batch = (
                (prepare_image("{}/{}".format(self.data_path,img), **self.img_settings), sa)
                for img, sa in data_batch
                )

        data_batch = (list(zip(imgs, [sa] * len(imgs))) for imgs, sa in data_batch)

        data_batch = [ex for img_list in data_batch for ex in img_list]

        imgs , sa = zip(*data_batch)
        
        """
        #print(features_data, np.array(labels_data))
        #print(len(self.features), len(self.features[0]))
        return np.array(features_data), np.array(labels_data)

    def get_img_data_gen(self):
        target_size = (self.img_settings["img_res"], self.img_settings["img_res"])
            
        color_mode = "grayscale" if self.img_settings["img_channels"] == 1 else "rgb"

        settings = {"directory": self.img_data_path, "x_col": "img_name", "y_col": "img_name", "class_mode": "raw",
                "batch_size": self.batch_size, "target_size": target_size, "color_mode": color_mode, 'shuffle': False}

        if(self.mode == 'train'): 
               data_gen = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, horizontal_flip=True)
               settings['shuffle'] = True
        else:
               data_gen = ImageDataGenerator(rescale=1. / 255)
                
        return data_gen.flow_from_dataframe(self.df, **settings)