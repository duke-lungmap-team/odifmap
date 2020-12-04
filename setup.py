from setuptools import setup

setup(
    name='ifmap',
    version='0.3.0b',
    packages=['ifmap'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='A Python library for ontology-driven object segmentation & classification '
                'in immunofluorescence microscopy images',
    install_requires=[
        'scikit-learn (>=0.23.1)',
        'scikit-image (>=0.17.2)',
        'opencv-python (>=4.3.0.36)',
        'scipy (>=1.5.3)',
        'numpy (>=1.19)',
        'matplotlib (>=3.3)',
        'Pillow (>=8.0.1)',
        'pandas (>=1.0.5)',
        'xgboost (>=1.1.1)',
        'cv2-extras @ git+https://github.com/whitews/cv2-extras.git',
        'ontospy (>=1.9.8.3)'
    ]
)
