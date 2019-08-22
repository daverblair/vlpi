#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:11:51 2019

@author: davidblair
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vlpi-pkg-davidblair",
    version="0.0.1",
    author="David Blair",
    author_email="david.blair@ucsf.edu",
    description="Python implementation of the variational latent phenotype model described in ____.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/____",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

