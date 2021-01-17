import numpy as np
import keras
import os
import albumentations as ab


class DataGenerator(keras.utils.Sequence):

    def __init__(self, subset, **kwargs):
        self.dim_in = kwargs['dim_in']
        self.dim_out = kwargs['dim_out']
        self.batch_size = kwargs['batch_size']
        self.list_IDs = [
            fl.replace('.jpg', '') for fl in os.listdir(
                os.path.join('data', kwargs['dataset'], subset)
            ) if fl.endswith('.jpg')
        ]
        self.items = kwargs['items']
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = np.load('data/' + ID + '.npy')
            y[i] = self.labels[ID]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
