
from tensorflow.keras.utils import Sequence
from image_utils import prepare_image
import numpy as np
import random


#IMG_TYPE=[".png"]
IMG_KEY_WORD = "img"

class DataSequence(Sequence):
    # used to create data generators to
    # avoid loading all the dataset to the memory
    # although this will result in a much longer training process

    def __init__(self, df, feature_cols, label_cols, data_paths, img_settings, 
                 feature_types=[str], label_types=[float],
                 batch_size=32, mode='train'):

        self.data_path = data_path
        self.df = df
        self.batch_size = batch_size
        self.mode = mode

        features = df[feature_cols].to_list()
        labels = df[label_cols].to_list()

        features = [feature_type[i](f) for i, f in enumerate(features)]
        labels = [label_type[i](f) for i, f in enumerate(labels)]
        
        self.f_type=["img" if IMG_KEY_WORD in f else "float" for f in df[feature_cols].columns]
        
        self.data = (features, labels)

        self.img_settings = img_settings

        self.on_epoch_end()

    def on_epoch_end(self):
        if not self.mode == 'train':
            return
        
        self.data=list(zip(self.data[0], self.data[1]))
        random.shuffle(self.data)                        # shuffle
        self.data=list(zip(*self.data))

    def __len__(self):
        return int(np.ceil(len(self.data[0]) / self.batch_size))

    def __getitem__(self, idx):
        bsz = self.batch_size
        
        # data_batch = self.data[bsz * idx: bsz * (idx + 1)]
        features = self.data[0][bsz * idx: bsz * (idx + 1)]
        labels = self.data[1][bsz * idx: bsz * (idx + 1)]
        
        f = features[:]
        load_img=False 
        
        for j in range(len(features[0])): 
            if self.f_type[j] == "img":
                load_img=True 
            for i in range(len(features)): 
                if load_img: 
                  f[i][j]=prepare_image("{}/{}".format(self.data_paths[j],img), **self.img_settings)
        """
        data_batch = (
                (prepare_image("{}/{}".format(self.data_path,img), **self.img_settings), sa)
                for img, sa in data_batch
                )

        data_batch = (list(zip(imgs, [sa] * len(imgs))) for imgs, sa in data_batch)

        data_batch = [ex for img_list in data_batch for ex in img_list]

        imgs , sa = zip(*data_batch)
        
        """
        return np.array(f), np.array(labels)





