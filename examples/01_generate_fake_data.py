from examples.utils import shapes
import matplotlib.pyplot as plt
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(
    count = 5,
    height = 250,
    width = 250
)
dataset_train.prepare()

print('Here are all the ids of the images we just generated: ')
print(dataset_train.image_ids)


print('Printing random background and color shapes.')
for id in dataset_train.image_ids:
    plt.imshow(dataset_train.load_image(id))
    plt.show()


# Static background color.
dataset_train = shapes.ShapesDataset()
dataset_train.load_shapes(
    count=5,
    height=250,
    width=250,
    background_color=(0, 0, 61)
)
dataset_train.prepare()

print('Here are all the ids of the images we just generated: ')
print(dataset_train.image_ids)

print('Printing random background and color shapes.')
for id in dataset_train.image_ids:
    plt.imshow(dataset_train.load_image(id))
    plt.show()


from numpy import array
dataset_make = shapes.ShapesDataset()
dataset_make.image_info = [{'bg_color': array([ 52, 218, 103]),
  'height': 250,
  'id': 0,
  'path': None,
  'shapes': [('square', (117, 225, 15), (156, 60, 23)),
             ('circle', (201, 160, 100), (145, 99, 56))],
  'source': 'shapes',
  'width': 250},
 {'bg_color': array([120, 140,   0]),
  'height': 250,
  'id': 1,
  'path': None,
  'shapes': [('triangle', (238, 4, 79), (21, 75, 50)),
             ('square', (161, 249, 183), (153, 20, 52))],
  'source': 'shapes',
  'width': 250},
 {'bg_color': array([ 31, 153,  61]),
  'height': 250,
  'id': 2,
  'path': None,
  'shapes': [('triangle', (32, 252, 60), (94, 185, 42))],
  'source': 'shapes',
  'width': 250},
 {'bg_color': array([151, 148, 198]),
  'height': 250,
  'id': 3,
  'path': None,
  'shapes': [('circle', (188, 231, 73), (74, 44, 22)),
             ('square', (90, 233, 124), (169, 92, 43)),
             ('circle', (132, 123, 209), (224, 125, 61))],
  'source': 'shapes',
  'width': 250},
 {'bg_color': array([ 69, 177, 237]),
  'height': 250,
  'id': 4,
  'path': None,
  'shapes': [('triangle', (56, 6, 132), (92, 28, 27)),
             ('triangle', (252, 76, 209), (212, 225, 24)),
             ('circle', (120, 134, 206), (24, 127, 43))],
  'source': 'shapes',
  'width': 250}]
