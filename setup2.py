from setuptools import setup, Extension

"""
    Calling
    $python setup.py build_ext --inplace
    will build the extension library in the current file.
"""
setup(
    name='myspectralsp',
    version='0.1.0',
    description="C-API - Kmeans algorithm",
    ext_modules=[
        Extension(
            'myspectralsp',
            ['spectral++.c'],
        ),
    ]
)
