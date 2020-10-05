
from tensorflow.keras.utils import Sequence
from image_utils import prepare_image
import numpy as np
import random



class DataSequence(Sequence):
    # used to create data generators to
    # avoid loading all the dataset to the memory
    # although this will result in a much longer training process

    def __init__(self, data_path, df, feature_col, label_col, img_settings, feature_type=str, label_type=float,
                 batch_size=32, mode='train', ):

        self.data_path = data_path
        self.df = df
        self.batch_size = batch_size
        self.mode = mode

        features = df[feature_col].to_list()
        labels = df[label_col].to_list()

        features = [feature_type(i) for i in features]
        labels = [label_type(i) for i in labels]

        self.data = list(zip(features, labels))

        self.img_settings = img_settings

        self.on_epoch_end()

    def on_epoch_end(self):
        if not self.mode == 'train':
            return

        random.shuffle(self.data)  # shuffle

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        bsz = self.batch_size
        data_batch = self.data[bsz * idx: bsz * (idx + 1)]

        data_batch = (
                (prepare_image("{}\\{}".format(self.data_path,img), **self.img_settings), sa)
                for img, sa in data_batch
                )

        data_batch = (list(zip(imgs, [sa] * len(imgs))) for imgs, sa in data_batch)

        data_batch = [ex for img_list in data_batch for ex in img_list]

        imgs , sa = zip(*data_batch)

        return np.array(imgs), np.array(sa)
