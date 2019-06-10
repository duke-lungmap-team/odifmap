from examples.utils import shapes
import matplotlib.pyplot as plt

dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(
    count = 1,
    height = 250,
    width = 250,
    background_color=(0, 0, 255)
)
dataset_train.prepare()

img1 = dataset_train.load_image(0)

# Static background color.
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(
    count=5,
    height=250,
    width=250,
    background_color=(0, 0, 61)
)
img2 = dataset_train.load_image(0)

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()

import cv2
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

from micap.utils import color_correction

#We'll make the darker colored image lighter by referencing the lighter colored image (index 0)
new = color_correction(
    [img1_hsv, img2_hsv],
    0
)

converted = [cv2.cvtColor(x, cv2.COLOR_HSV2RGB) for x in new]
plt.imshow(converted[0])
plt.show()

plt.imshow(converted[1])
plt.show()