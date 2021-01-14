import segmentation_models as sm
from data_generator import DataGenerator
from clr_callback import CyclicLR

BACKBONE = 'vgg19shrink'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Parameters
params = {
    'train': 'train',
    'validation': 'validation',
    'dim': (360, 640),
    'batch_size': 32,
    'n_classes': 3, #confidence, width kernel, height kernel
    'n_channels': 3, #for RGB image
    'shuffle': True
}

# Generators
training_generator = DataGenerator(preprocess_input, **params)
validation_generator = DataGenerator(preprocess_input, **params)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'SGD',
    loss=sm.losses.binary_crossentropy,
    metrics=[sm.metrics.recall, sm.metrics.precision],
    callbacks=[
        CyclicLR(
	        mode=config.CLR_METHOD,
	        base_lr=config.MIN_LR,
	        max_lr=config.MAX_LR,
	        step_size=config.STEP_SIZE*(trainX.shape[0]//config.BATCH_SIZE)
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
