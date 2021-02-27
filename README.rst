Table of Contents
~~~~~~~~~~~~~~~~~
 - `Introduction`_
 - `Data format`_
 - `Configuration`_
 - `Models`_
 - `Simple training pipeline`_
 - `Examples`_
 - `Citing`_
 - `License`_
 
Introduction
~~~~~~~~~~~
This project use the library of qubvel/segmentation_models most, aiming at most of CV tasks such as object detection(including the prediction of bboxes, keypoints, heatmap, ..., etc), appearance embedding learning, encoding/decoding of images and GAN. 

Data format
~~~~~~~~~~~
The data is stored in .xlsx file with single sheet for non-series model and with multiple sheets for time-series model. The format is like

+----------+---------------------------------------------------------+
| Column 1 | Column 2                                                |
+==========+=========================================================+
| Foo      | Put two (or more) spaces as a field separator.          |
+----------+---------------------------------------------------------+
| Bar      | Even very very long lines like these are fine, as long  |
|          | as you do not put in line endings here.                 |
+----------+---------------------------------------------------------+
| Qux      | This is the last line.                                  |
+----------+---------------------------------------------------------+

Configuration
~~~~~~~~~~~

Models
~~~~~~~~~~~

Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    import segmentation_models as sm

    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # load your data
    x_train, y_train, x_val, y_val = load_data(...)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(BACKBONE, encoder_weights='imagenet')
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=16,
       epochs=100,
       validation_data=(x_val, y_val),
    )

Same manipulations can be done with ``Linknet``, ``PSPNet`` and ``FPN``. For more detailed information about models API and  use cases `Read the Docs <https://segmentation-models.readthedocs.io/en/latest/>`__.

Examples
~~~~~~~~
Models training examples:
 - [Jupyter Notebook] Binary segmentation (`cars`) on CamVid dataset `here <https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb>`__.
 - [Jupyter Notebook] Multi-class segmentation (`cars`, `pedestrians`) on CamVid dataset `here <https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb>`__.

Citing
~~~~~~~~

.. code::

    @misc{Yakubovskiy:2019,
      Author = {Pavel Yakubovskiy},
      Title = {Segmentation Models},
      Year = {2019},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/qubvel/segmentation_models}}
    } 

License
~~~~~~~
Project is distributed under `MIT Licence`_.

.. _CHANGELOG.md: https://github.com/qubvel/segmentation_models/blob/master/CHANGELOG.md
.. _`MIT Licence`: https://github.com/qubvel/segmentation_models/blob/master/LICENSE
