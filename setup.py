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
        'git+https://github.com/whitews/cv2-extras.git',
        'matplotlib (>=3.0)',
        'numpy (>=1.16)',
        'opencv-python (>=4.1)',
        'pandas (>=0.24)',
        'Pillow (>=6.0.0)',
        'scikit-image (>=0.15)',
        'scikit-learn (>=0.20.3)',
        'scipy (>=1.2)',
        'scipy (>=1.2.1)',
        'xgboost (>=0.82)'
    ]
)
