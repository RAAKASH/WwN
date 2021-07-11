# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:13:29 2021

@author: ECGC
"""

  
from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extensions = [Extension("*", ["*.pyx"])]

setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)