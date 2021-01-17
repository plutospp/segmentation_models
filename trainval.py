import segmentation_models as sm
from data_generator import DataGenerator
from clr_callback import CyclicLR
from unet import UNet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", type=str, help="training item")
args = parser.parse_args()

# Generators
training_generator = DataGenerator(subset='training', **args.params)
validation_generator = DataGenerator(subset='validation', **args.params)

# define model
model = UNet(pretrained_weights='lp_detect.h5')
model.compile(
    'SGD',
    loss=sm.losses.binary_crossentropy,
    metrics=[sm.metrics.recall, sm.metrics.precision],
    callbacks=[
        CyclicLR(
	        mode='triangular',
	        base_lr=0.00001,
	        max_lr=0.001,
	        step_size=params['steps']
        )
    ]
)

# fit model
model.fit_generator(
    generator=training_generator,
    validation_data=validation_generator,
    use_multiprocessing=True,
    workers=6
)
