#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:11:51 2019

@author: davidblair
"""

###

import setuptools
import re

version = re.search(
    '^__version__\s*=\s*"(.*)"',
    open('vlpi/vLPI.py').read(),
    re.M).group(1)



with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="vlpi",
    version=version,
    author="David Blair",
    author_email="david.blair@ucsf.edu",
    description="Python implementation of the variational latent phenotype model described in Blair et al..",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daverblair/vlpi",
    package_data={'vlpi': ['data/ICDData/*.txt']},
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'torch==1.5.1',
        'pyro-ppl>=1.3.1',
        'numpy>=1.19.0',
        'pandas>=1.0.5',
        'scipy>=1.5.2',
        'scikit-learn>=0.22.1',
        'typing',
        'unidecode'
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
