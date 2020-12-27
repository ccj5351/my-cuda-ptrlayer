# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: setup.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 11-10-2019
# @last modified: Fri 11 Oct 2019 01:44:49 PM EDT

"""
#> see https://pytorch.org/tutorials/advanced/cpp_extension.html 
"""

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
        ext_modules=[
            cpp_extension.CppExtension('lltm_cpp', [
                'lltm.cpp'
                ])
            ],
        cmdclass={
            'build_ext': cpp_extension.BuildExtension
            }
    )
