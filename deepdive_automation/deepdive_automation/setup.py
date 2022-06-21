from setuptools import setup

setup(
    name='deepdive_automation',
    version='0.1.0',    
    description='A Python package to automate deep dive methodologies',
    url='https://github.com/Earthshot-Labs/model_utilities/deepdive_automation',
    author='Earthshot Science Team',
    packages=['deepdive_automation'],
    install_requires=['pandas',
                      'numpy', 
                      'geopandas',
                      'matplotlib',
                      'seaborn',
                      'ee',
                      'scipy'                    
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)