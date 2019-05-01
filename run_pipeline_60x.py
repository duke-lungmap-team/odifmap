import os
import numpy as np
from pipeline import pipeline
from glob import glob
from PIL import Image
import cv2_extras as cv2x

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

cell_radius = 17 * 3
cell_size = np.pi * (cell_radius ** 2)

seg_config = [
    {
        'type': 'color',
        'args': {
            'blur_kernel': (51, 51),
            'min_size': 3 * cell_size,
            'max_size': None,
            'colors': ['green', 'cyan', 'red', 'violet', 'yellow']
        }
    },
    {
        'type': 'saturation',
        'args': {'blur_kernel': (71, 71), 'min_size': 12 * cell_size, 'max_size': None}
    },
    {
        'type': 'saturation',
        'args': {'blur_kernel': (53, 53), 'min_size': 3 * cell_size, 'max_size': 45 * cell_size}
    },
    {
        'type': 'saturation',
        'args': {'blur_kernel': (35, 35), 'min_size': 3 * cell_size, 'max_size': 45 * cell_size}
    },
    {
        'type': 'saturation',
        'args': {'blur_kernel': (17, 17), 'min_size': 3 * cell_size, 'max_size': 45 * cell_size}
    }
]

image_set_dir = 'mm_e16.5_60x_sox9_sftpc_acta2'

# make our 'tmp' directory for caching trained & tested pipeline instances
if not os.path.isdir('tmp'):
    os.mkdir('tmp')


output_path = os.path.join(
    'tmp',
    '_'.join([image_set_dir, 'pipeline'])
)
image_set_path = os.path.join('data', image_set_dir)

image_paths = glob(os.path.join(image_set_path, '*.tif'))

tmp_image = Image.open(image_paths[2])
tmp_image = np.asarray(tmp_image)
tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2HSV)

# and pipeline test steps
candidate_contours = pipeline.generate_structure_candidates(
    tmp_image,
    seg_config,
    filter_min_size=3 * cell_size,
    plot=True
)
cv2x.plot_contours(tmp_image, candidate_contours)
# test_data_processed = pipeline.process_test_data(test_img_hsv, candidate_contours)

# plot functions
# pipeline.plot_test_results(test_img_hsv, candidate_contours, pred_results, output_path)

# optional cell segmentation
# utils.process_structures_into_cells(
#     test_img_hsv,
#     os.path.join(output_path, 'regions'),
#     candidate_contours,
#     plot=False
# )
