from setuptools import setup

setup(
    name='micap',
    version='0.2.1',
    packages=['micap'],
    license='BSD 2-Clause License',
    long_description=open('README.md').read(),
    author='Scott White',
    description='A Python library for segmentation and classification of objects in microscopy images',
    install_requires=[
        'scikit-learn (>=0.20.3)',
        'scikit-image (>=0.15)',
        'opencv-python (>=4.1)',
        'scipy (>=1.2)',
        'numpy (>=1.16)',
        'matplotlib (>=3.0)',
        'Pillow (>=6.2.2)',
        'pandas (>=0.24)',
        'xgboost (>=1.1)',
        'cv2-extras @ git+https://github.com/whitews/cv2-extras.git'
    ]
)
