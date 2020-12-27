# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: lltm.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 11-10-2019
# @last modified: Sat 12 Oct 2019 11:09:57 AM EDT

""" 
Once your extension is built, you can simply import it in Python, 
using the name you specified in your setup.py script. 
Just be sure to import torch first, as this will resolve some symbols 
that the dynamic linker must see;
"""
import math
from torch import nn
from torch.autograd import Function
import torch
import time

# cuda model
from build.lib import lltm_cuda

torch.manual_seed(42)


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        outputs = lltm_cuda.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)
        return new_h, new_cell
    
    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        outputs = lltm_cuda.backward(
                grad_h.contiguous(), grad_cell.contiguous(), 
                *ctx.saved_tensors
                )
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell



class LLTM(nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(
                torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    
    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)


if __name__ == "__main__":
    
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")  # device object representing GPU
    
    batch_size = 16
    input_features = 32
    state_size = 128
    
    X = torch.randn(batch_size, input_features, device=device)
    h = torch.randn(batch_size, state_size, device=device)
    C = torch.randn(batch_size, state_size, device = device)
    rnn = LLTM(input_features, state_size).to(device)
    
    # Force CUDA initialization
    new_h, new_C = rnn(X, (h, C))
    (new_h.sum() + new_C.sum()).backward()

    #forward_min = math.inf
    forward_time = 0
    #backward_min = math.inf
    backward_time = 0
    
    runs = 100000
    for _ in range(runs):
        rnn.zero_grad()
        
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        elapsed = time.time() - start
        #forward_min = min(forward_min, elapsed)
        forward_time += elapsed
        
        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        elapsed = time.time() - start
        #backward_min = min(backward_min, elapsed)
        backward_time += elapsed
    
    print('Forward: {:.4f} seconds | Backward {:.4f} seconds'.format(
        forward_time, backward_time))
