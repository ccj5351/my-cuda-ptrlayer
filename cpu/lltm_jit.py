# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: lltm.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 11-10-2019
# @last modified: Fri 11 Oct 2019 05:18:39 PM EDT

""" 
Once your extension is built, you can simply import it in Python, 
using the name you specified in your setup.py script. 
Just be sure to import torch first, as this will resolve some symbols 
that the dynamic linker must see;
"""
import math
import torch
import time

""" method 1: Building with setuptools 
# Our module!
from build.lib import lltm_cpp
"""

""" JIT Compiling Extensions: just in time, JIT """
from torch.utils.cpp_extension import load
lltm_cpp = load(name="lltm_cpp", sources=["lltm.cpp"], 
        #verbose = False
        verbose = True
        )


class LLTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cpp.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cpp.backward(
            grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
        d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


if __name__ == "__main__":
    
    
    assert torch.cuda.is_available()
    #device = torch.device("cuda:0")  # device object representing GPU
    device = torch.device("cpu:0")  # device object representing CPU
    
    batch_size = 16
    input_features = 32
    state_size = 128

    X = torch.randn(batch_size, input_features, device=device)
    h = torch.randn(batch_size, state_size, device=device)
    C = torch.randn(batch_size, state_size, device = device)

    rnn = LLTM(input_features, state_size).to(device)

    forward = 0
    backward = 0

    for _ in range(100000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        torch.cuda.synchronize()
        
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
        
    print('Forward: {:.3f} seconds | Backward {:.3f} seconds'.format(
            forward, 
            backward))
