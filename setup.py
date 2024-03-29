from setuptools import setup

setup(
    name='model_utilities', #model_utilties
    version='0.2.0',
    description='A Python package with utility functions for ecosystem analysis and modeling',
    url='https://github.com/Earthshot-Labs/model_utilities',
    author='Earthshot Science Team',
    packages=['model_utilities'], #model_utilities
    install_requires=['pandas',
                      'numpy', 
                      'geopandas',
                      'matplotlib',
                      'seaborn',
                      'ee',
                      'scipy',
                      'GDAL',
                      'google-cloud-storage',
                      'scikit-learn',
                      'tqdm',
                      'shutil'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.8'
    ],
)
