from segmentation_models.losses import *
from data_generator import DataGenerator
from clr_callback import CyclicLR
from keras.models import model_from_yaml
import json
import keras
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", type=str, help="training item")
args = parser.parse_args()

params = json.load(open('params/'+args.params+'.json'))

# Generators
training_data = DataGenerator(subset='training', **params)
#validation_generator = DataGenerator(subset='validation', **params)

# define model
model = model_from_yaml(open('models/'+params["model"]+'.yaml'))
model.load_weights('models/'+params["model"]+'.h5', by_name=True)
VARS = vars()
model.compile('SGD', loss={
    k: VARS[v] for k, v in params['vars_loss'].items()
})

# fit model
model.fit(
    training_data,
    epochs=10000,
    #validation_data=validation_generator,
    #use_multiprocessing=True,
    #workers=6,
    callbacks=[
        CyclicLR(
	        mode='triangular',
	        base_lr=0.000001,
	        max_lr=0.01,
	        step_size=params['steps']
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='models/'+params['model']+'.h5',
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True
        )
    ]
)
