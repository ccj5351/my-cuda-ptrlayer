# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: jit.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-10-2019
# @last modified: Sat 12 Oct 2019 12:08:47 AM EDT

from torch.utils.cpp_extension import load
lltm_cuda = load(
            'lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)
