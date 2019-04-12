import numpy as np
from PIL import Image
import os
from pipeline import utils, pipeline
import json

# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

image_dir = 'data/mm_e16.5_20x_sox9_sftpc_acta2/light_color_corrected'
test_img_name = '2015-04-029_20X_C57Bl6_E16.5_LMM.14.24.4.46_SOX9_SFTPC_ACTA2_002.tif'
test_img_path = os.path.join(image_dir, test_img_name)

tmp_img = Image.open(test_img_path)
# noinspection PyUnresolvedReferences
test_img_hsv = cv2.cvtColor(np.asarray(tmp_img), cv2.COLOR_RGB2HSV)
test_img_hsv = test_img_hsv[0:1000, 0:1000]
test_save_dir = os.path.join(
    'trad_seg_testing2',
    os.path.basename(test_img_path)
)

cell_radius = 16
cell_size = np.pi * (cell_radius ** 2)

seg_config = [
    {
        'type': 'color',
        'args': {
            'blur_kernel': (17, 17),
            'min_size': 3 * cell_size,
            'max_size': None,
            'colors': ['green', 'cyan', 'red', 'violet', 'yellow']
        }
    }
]

all_color_contours = pipeline.generate_structure_candidates(
    test_img_hsv,
    seg_config,
    filter_min_size=2 * cell_size,
    process_residual=False,
    plot=True
)
print("%d color cell candidates found" % len(all_color_contours))

structures_with_cells = utils.process_structures_into_cells(
    test_img_hsv,
    test_save_dir,
    all_color_contours,
    ['green', 'cyan', 'blue'],
    cell_size,
    plot=False
)

json_sc = {
    test_img_name: {'regions': structures_with_cells}
}

json_string = json.dumps(
    json_sc,
    indent=2,
    default=utils.numpy_json_converter
)

out_file = open(os.path.join(test_save_dir, 'cell_regions.json'), 'w')
out_file.write(json_string)
out_file.close()
