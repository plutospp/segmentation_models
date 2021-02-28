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

+--------+--------+--------+--------+--------+--------+
|  key1  |  key1  |  key2  |  key2  |  key2  |  key2  |
+========+========+========+========+========+========+
| value1 | value2 | value1 | value2 | value3 | value4 |
+--------+--------+--------+--------+--------+--------+

The keys are strings and values could be file paths or numbers. One key may correspond to several values.

Configuration
~~~~~~~~~~~
The confiuration .json files stored in folder "config" contains informations how the program interpretates the data recorded in the .xlsx file.

.. code:: json

    {
        "hello": "world",
        "this": {
            "can": {
                "be": "nested"
            }
        }
    }
    
The values corresponding to keys of image or array are the string of file path; that of keypoints are 2n numbers(x and y of n points); that of bboxes are 4n numbers(xmin, ymin xmax, ymax of n boxes)

Models
~~~~~~~~~~~
The keras models are stored at the folder "models" which contains the .yaml for the model graph and .h5 for the model weights. The names of inputs and outputs should be consistant with the corresponding configuration.

Simple training pipeline
~~~~~~~~~~~~~~~~~~~~~~~~
After setting the model(graph and pretrained weights), configuration and generation of data, just execute

.. code:: python
    $ python3 trainval.py --config xxx

Examples
~~~~~~~~
Face Analysis:
 - [Dataset]  `here <https://www.kaggle.com/c/facial-keypoints-detection/data>`__.

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
