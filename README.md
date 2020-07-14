# microscopy-analysis-pipeline (micap)
A Python library for segmentation and classification of objects in microscopy images.

## Environment
To match the environment requirements needed to complete the examples, please install all
dependencies listed in the `requirements.txt` file.

```
pip install -r requirements.txt
```

## Usage

### Importing `micap` modules

The `micap` package contains 2 modules used for executing the analysis pipeline: `utils` 
and `pipeline`:

    from micap import utils, pipeline

The `utils` module contains functions related to pre- and post-processing, including 
loading the training data and processing the pipeline's structure contours into cell
contours.

### Training Data

The set of images for training must be in a directory containing a 'regions.json' file. 
This regions file has keys of the image file names in the image set, and the value for 
each image is a dictionary of class labels, and the value of those labels is a list of
segmented polygon regions. An example of the JSON format is shown below:

    {
      "_0.tif": {
        "full": false, 
        "regions": {
          "square": [
            [[1228, 970], [1228, 1064], [1322, 1064], [1322, 970]], 
            [[0, 287], [0, 395], [84, 395], [84, 287]], 
            [[225, 1244], [225, 1338], [319, 1338], [319, 1244]]
          ],
          "circle": [
            ...
          ]
        }
      }
    }

At the top level, the keys shall be the names of image files. The value shall be 
another JSON object which includes the 2 keys "full" and "regions". 

The "full" value is a boolean and indicates whether the regions provide full coverage 
over the entire training image. Full coverage means that all the regions of interest 
for every structure in the image is included. This is used to automatically detect 
"background" in the training data set. For fully covered training images, the remaining 
areas not specified by the regions are divided into sub-regions representing a 
"background" class.

The "regions" value shall be a JSON object where the keys are the names of the structure
classes. The value of every structure class key is a list of regions. Each region in the list
is a list of x and y coordinates representing the vertices of the region's contour.There 
are no restrictions on the number of region class labels or their names.

### Training the model

First, load the image set data using the path to the training image set directory 
containing the training images and the regions.json file:

    training_data = utils.get_training_data_for_image_set(image_set_path)

Next, process the training data to create the feature metrics used to train the model 
using the `pipeline` method `process_training_data`:

    training_data_processed = pipeline.process_training_data(training_data)

Finally, run `fit` to train the model, returning the model and lookup table for the 
category labels:

    xgb_model, categories = pipeline.fit(training_data_processed)

### Create a segmentation configuration

Multiple segmentation stages can be run to extract structures of varying sizes, different 
color combinations, and color intensity (saturation levels). The configuration of 
segmentation stages is created using a list of dictionaries, where each stage will 
be processed in the order they appear in the list. There are two types of segmentation
stages, one is color-based and the other is based on saturation levels. Each stage is
a dictionary containing two keywords: `type` and `args`. The `type` key will have the text
value of "color" or "saturation" and specifies the type of segmentation to be performed. 
The `args` key contains the arguments, or options, for that stage in the form of another
dictionary. The `args` dictionary 

    seg_config = [
        {
            'type': 'color',
            'args': {
                'blur_kernel': (15, 15),
                'min_size': 3 * cell_size,
                'max_size': None,
                'colors': ['green', 'cyan', 'red', 'violet']
            }
        },
        {
            'type': 'saturation',
            'args': {'blur_kernel': (63, 63), 'min_size': 3 * cell_size, 'max_size': None}
        },
        {
            'type': 'saturation',
            'args': {'blur_kernel': (31, 31), 'min_size': 3 * cell_size, 'max_size': None}
        }
    ]