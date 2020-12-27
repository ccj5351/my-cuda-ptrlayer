# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: setup.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-10-2019
# @last modified: Sat 12 Oct 2019 10:07:37 AM EDT

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='lltm_cuda',
        ext_modules=[
            CUDAExtension('lltm_cuda', [
                'lltm_cuda.cpp',
                'lltm_cuda_kernel.cu',
                ]),
            ],
        cmdclass={
            'build_ext': BuildExtension
            }
        )
