from setuptools import setup

setup(
    name='modelUtilities',
    version='0.1.0',    
    description='Model Feature and Data Organizing Modules',
    url='',
    author='Meghan Blumstein',
    author_email='meghan@earthshot.eco',
    license='',
    packages=['modelUtilities'],
    install_requires=['rasterio',
                      'pandas',
                      'numpy',
                      'geopandas',
                      'netCDF4'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: MacOS :: MacOS X',        
        'Programming Language :: Python :: 3.8',
    ],
)
