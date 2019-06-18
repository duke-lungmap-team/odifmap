import os
import numpy as np
from micap import utils, pipeline
import pickle

cell_radius = 16
cell_size = np.pi * (cell_radius ** 2)

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
    },
    {
        'type': 'saturation',
        'args': {'blur_kernel': (15, 15), 'min_size': 3 * cell_size, 'max_size': None}
    },
    {
        'type': 'color',
        'args': {
            'blur_kernel': (15, 15),
            'min_size': 3 * cell_size,
            'max_size': None,
            'colors': ['green', 'yellow', 'red']
        }
    },
]

image_set_dir = 'examples/images'
image_set_path = image_set_dir
output_path = os.path.join(
    'examples',
    'images',
    'tmp'
)

# make our 'tmp' directory for caching trained & tested pipeline instances
if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)


try:
    # load pickled model
    f = open(os.path.join(output_path, 'xgb_model.pkl'), 'rb')
    pck = pickle.load(f)
    f.close()

    xgb_model = pck['model']
    categories = pck['categories']
    test_img_hsv = pck['test_img_hsv']
except FileNotFoundError:
    # get training data
    training_data = utils.get_training_data_for_image_set(image_set_path)
    # remove an image from training data to use for predict testing
    test_img_name = '_0.tif'
    test_data = training_data.pop(test_img_name)
    test_img_hsv = test_data['hsv_img']

    # train model
    training_data_processed = pipeline.process_training_data(training_data)
    xgb_model, categories = pipeline.fit(training_data_processed)

    # pickle the xgb model and categories here
    pck = {
        'model': xgb_model,
        'categories': categories,
        'test_img_hsv': test_img_hsv
    }
    f = open(os.path.join(output_path, 'xgb_model.pkl'), 'wb')
    pickle.dump(pck, f)
    f.close()

# and pipeline test steps
candidate_contours = pipeline.generate_structure_candidates(
    test_img_hsv,
    seg_config,
    filter_min_size=3 * cell_size,
    dog_factor=7,
    process_residual=False,
    predict_model=xgb_model,
    categories=categories,
    plot=False,
    use_signal_mask=False
)
test_data_processed = pipeline.process_test_data(test_img_hsv, candidate_contours)
if test_data_processed.shape[0]>0:
    pred_results = pipeline.predict(test_data_processed, xgb_model, categories)

    # plot functions
    pipeline.plot_test_results(
        test_img_hsv,
        candidate_contours,
        pred_results,
        output_path
    )
