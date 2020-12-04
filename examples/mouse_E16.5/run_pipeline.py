import os
import json
import numpy as np
import pickle
from ifmap import utils, pipeline, onto_utils
from ifmap.ontology import onto as lung_ontology

cell_radius = 16
cell_size = np.pi * (cell_radius ** 2)

image_set_path = 'images'
output_path = os.path.join(
    'tmp',
    '_'.join([image_set_path, 'output'])
)

image_metadata = json.load(
    open(os.path.join(image_set_path, "metadata.json"), 'r')
)
seg_config = onto_utils.build_seg_config(
    lung_ontology,
    probes=image_metadata['probes'],
    probe_colors=image_metadata['colors'],
    cell_size=cell_size
)

# make our 'tmp' directory for caching trained & tested pipeline instances
if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

training_data = utils.get_training_data_for_image_set(image_set_path)

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
    test_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_001.tif'
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
    plot=False
)
test_data_processed = pipeline.process_test_data(test_img_hsv, candidate_contours)
pred_results = pipeline.predict(test_data_processed, xgb_model, categories)

# plot functions
pipeline.plot_test_results(
    test_img_hsv,
    candidate_contours,
    pred_results,
    output_path
)

# optional cell segmentation
# utils.process_structures_into_cells(
#     test_img_hsv,
#     os.path.join(output_path, 'regions'),
#     candidate_contours,
#     cell_color_list=['green', 'cyan'],
#     max_cell_area=1.0*cell_size,
#     plot=True
# )
