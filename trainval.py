import segmentation_models as sm
from data_generator import DataGenerator
from clr_callback import CyclicLR
from unet import UNet


# Parameters
params = {
    'train': 'train',
    'validation': 'validation',
    'dim': (360, 640),
    'batch_size': 32,
    'n_classes': 3, #confidence, width kernel, height kernel
    'n_channels': 3, #for RGB image
    'shuffle': True,
    'steps': 500,
    'name': 'lp_detect'
}

# Generators
training_generator = DataGenerator(preprocess_input=None, **params)
validation_generator = DataGenerator(preprocess_input=None, **params)

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
