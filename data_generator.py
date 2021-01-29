import numpy as np
import keras
import os
import albumentations as ab
import imagehash
import csv
import ast
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)


def get_imagehash(self, img, hash_type, hash_size):
    hash_fn = getattr(imagehash, hash_type)
    return hash_fn(hash_size = hash_size).hash

class DataGenerator(keras.utils.Sequence):

    def __init__(self, subset, **kwargs):
        self.dataset = kwargs['dataset']
        self.list_IDs = [
            fl.replace('.csv', '') for fl in os.listdir(
                os.path.join('datasets', self.dataset, subset)
            ) if fl.endswith('.csv')
        ]
        self.globals = kwargs['global']
        self.locals = kwargs['local']
        self.shuffle = kwargs['shuffle']
        self.batch_size = kwargs['batch_size']
        self.aug = __get_augmentor(**kwargs['data_augmentations'])
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __processing_global(self, data, params):
        Xs = []
        for ki, vi in data['inputs'].items():
            X = np.empty((self.batch_size, *vi))
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = np.load('data/' + ID + '.npy')
        Ys = []
        for ko, vo in data['outputs'].items():
            Y = np.empty((self.batch_size, *vo))
            for i, ID in enumerate(list_IDs_temp):
                Y[i,] = np.load('data/' + ID + '.npy')
        return Xs, Ys

    def __processing_local(self, data, params):
        Xs = []
        for ki, vi in data['inputs'].items():
            X = np.empty((self.batch_size, *vi))
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = np.load('data/' + ID + '.npy')
        Ys = []
        for ko, vo in data['outputs'].items():
            Y = np.empty((self.batch_size, *vo))
            for i, ID in enumerate(list_IDs_temp):
                Y[i,] = np.load('data/' + ID + '.npy')
        return Xs, Ys

    def __data_generation(self, list_IDs_temp):
        data_global = []
        data_local = []
        for ID in list_IDs_temp:
            with open(ID+'.csv') as csvfl:
                lines = [
                    [
                        ast.literal_eval(x) for x in l
                    ] for l in csv.reader(csvfl)
                ]
            data_global.append(lines[0])
            data_local.append(lines[1:])
        Xs_global, Ys_global = self.__processing_global(data_global, self.globals)
        Xs_local, Ys_local = self.__processing_local(data_local, self.locals)
        return Xs_global + Xs_local, Ys_global + Ys_local
    
    def __get_augmentor(self, **kwargs):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=0.5)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        Xs, Ys = self.__data_generation(list_IDs_temp)
        return Xs, Ys
