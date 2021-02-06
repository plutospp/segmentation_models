import numpy as np
import keras
import os
import cv2
import imagehash
import csv
import pandas as pd
from itertools import groupby
from albumentations import (
    BboxParams, HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, MultiplicativeNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, KeypointParams, RandomBrightnessContrast, Flip, HorizontalFlip, OneOf, Compose
)


def get_imagehash(self, img, hash_type, hash_size):
    hash_fn = getattr(imagehash, hash_type)
    return hash_fn(hash_size = hash_size).hash

class DataGenerator(keras.utils.Sequence):

    def __init__(self, subset, **kwargs):
        self.subset = subset
        self.dataset = kwargs['dataset']
        self.list_IDs = [
            fl.replace('.xlsx', '') for fl in os.listdir(
                os.path.join('datasets', self.dataset, self.subset)
            ) if fl.endswith('.xlsx')
        ]
        self.global_vars = kwargs.get('global_vars', {})
        self.object_vars = kwargs.get('object_vars', {})
        self.shuffle = kwargs['shuffle']
        self.batch_size = kwargs['batch_size']
        self.object_hash = kwargs.get('object_hash', 0)
        preprocess_method = kwargs.get('preprocess', 'caffe')
        if preprocess_method=='caffe':
            self.preprocess = lambda x: x-127
        else:
            self.preprocess = lambda x: x/255.
        self.augment_params = self.__get_augmentation()
        self.on_epoch_end()      

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __parse_global(self, data, params, series_num, tag):
        data_keys = data[0]
        data_vals = data[1]
        vars_dict = {}
        for k, v in params.items():
            vals = []
            for kd, vd in zip(data_keys, data_vals):
                if k in kd:
                    vals.append(vd)
            if 'image' in v:
                var = cv2.imread(
                    os.path.join('datasets', self.dataset, vals[0]),
                    cv2.IMREAD_COLOR
                )
                tp = 'image'
            elif 'mask' in v:
                _, mats = cv.imreadmulti(os.path.join('datasets', self.dataset, vals[0]))
                var = np.concatenate([mat[...,0] for mat in mats], axis=2)/255.
                tp = 'mask'
            elif 'bboxes' in v:
                var = [box+['bbox'] for box in np.reshape(vals, (-1,4)).tolist()]
                tp = 'bboxes'
            elif 'keypoints' in v:
                var = [keypoint+['keypoint'] for keypoint in np.reshape(vals, (-1,2)).tolist()]
                tp = 'keypoints'
            var_name = '--'.join([tag, str(series_num), k])
            vars_dict.update({var_name: {'value': var, 'type': tp}})
        return vars_dict

    def __parse_local(self, data, params, series_num, tag):
        vars_dict = {}
        return vars_dict
    
    def __get_augmentation(self):
        return [
            #HorizontalFlip(p=0.5),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
                MultiplicativeNoise()
            ], p=0.25),
            #OneOf([
            #    MotionBlur(p=0.2),
            #    MedianBlur(blur_limit=3, p=0.1),
            #    Blur(blur_limit=3, p=0.1),
            #], p=0.2),
            #ShiftScaleRotate(
                #shift_limit=0.0625,
                #scale_limit=0.2,
            #    rotate_limit=10,
            #    p=0.25
            #),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                RandomBrightnessContrast(),
                #OpticalDistortion()
            ], p=0.3),
            #HueSaturationValue(p=0.3),
        ]

    def __data_parse(self, ID):
        vars_dict = {}
        data = pd.read_excel(
            os.path.join('datasets', self.dataset, self.subset, ID+'.xlsx'),
            sheet_name=None
        )
        for series_num, (sheet_name, df) in enumerate(data.items()):
            lines = [
                [
                    float(x) if x.replace('.', '').replace('-', '').isnumeric() else x for x in l.split(',')
                ][1:] for l in df.to_csv().split('\n')
            ][:-1]
            vars_dict.update(self.__parse_global(lines[:2], self.global_vars, series_num, 'global_vars'))
            #vars_dict.update(self.__parse_objects(lines[2:], self.objects_vars, series_num, 'objects_vars'))
        transformed = {k: v['value'] for k, v in vars_dict.items()}
        transform = Compose(
            self.augment_params,
            additional_targets = {k: v['type'] for k, v in vars_dict.items()},
            keypoint_params = KeypointParams(
                format='xy', remove_invisible=False),
            bbox_params = BboxParams(
                format='albumentations', min_visibility=0.1),
            p=0.5
        )
        transformed = transform(image=np.array([[0,0,0]], dtype='uint8'), bboxes=[], keypoints=[], **{k: v['value'] for k, v in vars_dict.items()})
        for k, v in transformed.items():
            if k in ['image', 'bboxes', 'keypoints']:
                continue
            vars_dict[k].update({'value': v})
        return vars_dict
        
    def __gen_global(self, vs):
        var_dict = {}
        for var_name, var_group in groupby(vs, lambda v: v['name']):
            var_series = sorted(list(var_group), key=lambda x: x['series'])
            var_dict.update({
                var_name: {
                    'type': var_series[0]['type'],
                    'field': var_series[0]['field'],
                    'series': [var['value'] for var in var_series]
                }
            })
        input_dict = {}
        output_dict = {}
        for var_k, var_v in var_dict.items():
            if var_v['type']=='bboxes':
                bboxes_arr = []
                bboxes_hash = []
                for num_series, bboxes in enumerate(var_v['series']):
                    bboxes = [bbox[:-1] for bbox in bboxes]
                    bboxes_arr.append(np.reshape(bboxes, [1,1,-1]))
                    if 'hash' in self.global_vars[var_k]:
                        hash_param = self.global_vars[var_k]['hash']
                        image = var_dict[hash_param['image']]['series'][num_series]
                        h, w, _ = image.shape
                        bbox_hash = []
                        for bbox in bboxes:
                            x1, y1, x2, y2 = bbox
                            bbox_hash.extend(
                                get_imagehash(
                                    image[
                                        int(max(h*y1,0)):int(min(h*y2,h)),
                                        int(max(w*x1,0)):int(min(w*x2,w)), :
                                    ],
                                    hash_param['type'], hash_param['size']
                                )
                            )
                        bboxes_hash.append(np.reshape(bbox_hash, (1,1,-1)))
                output_dict.update({
                    var_k: np.stack(bboxes_arr) if len(bboxes_arr)>1 else bboxes_arr[0]
                })
                if bboxes_hash:
                    output_dict.update({
                        var_k+'_hash': np.stack(bboxes_hash) if len(bboxes_hash)>1 else bboxes_hash[0]
                    })
            elif var_v['type']=='keypoints':
                keypoints_arr = []
                for num_series, keypoints in enumerate(var_v['series']):
                    keypoints = [kpt[:-1] for kpt in keypoints]
                    keypoints_arr.append(np.reshape(keypoints, [1,1,-1]))
                output_dict.update({
                    var_k: np.stack(keypoints_arr) if len(keypoints_arr)>1 else keypoints_arr[0]
                })
            elif var_v['type'] in ['image', 'mask']:
                values = var_v['series']
                var_arr = np.stack(values) if len(values)>1 else values[0]
                if var_v['type']=='image':
                    var_arr = self.preprocess(var_arr)
                if self.global_vars[var_k]['io']=='in':
                    input_dict.update({var_k: var_arr})
                else:
                    output_dict.update({var_k: var_arr})
        return input_dict, output_dict
    
    def __gen_objects(self, vs):
        pass
            
    def __data_generation(self, ID):
        vars_dict = self.__data_parse(ID)
        vars_list = []
        for K, V in vars_dict.items():
            keys = K.split('--')
            vars_list.append({
                'field': keys[0],
                'series': int(keys[1]),
                'name': keys[2],
                'type': V['type'],
                'value': V['value']
            })
        global_Xs, global_Ys = self.__gen_global([var for var in vars_list if var['field']=='global_vars'])
        #objects_Ys = self.__gen_objects([var for var in vars_list if var['field']=='object_vars'])
        return global_Xs, global_Ys # + objects_Ys

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        Xs_batch = {}
        Ys_batch = {}
        for ID in list_IDs_temp:
            Xs, Ys = self.__data_generation(ID)
            for kx, vx in Xs.items():
                if kx not in Xs_batch:
                    Xs_batch.update({kx: [vx]})
                else:
                    Xs_batch.update({kx: Xs_batch[kx]+[vx]})
            for ky, vy in Ys.items():
                if ky not in Ys_batch:
                    Ys_batch.update({ky: [vy]})
                else:
                    Ys_batch.update({ky: Ys_batch[ky]+[vy]})
        Xs_out = {kx: np.stack(vx) for kx, vx in Xs_batch.items()}
        Ys_out = {ky: np.stack(vy) for ky, vy in Ys_batch.items()}
        return Xs_out, Ys_out
