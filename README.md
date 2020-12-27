# C++/CUDA Extensions in PyTorch

> see: Copied from https://github.com/pytorch/extension-cpp, by [Peter Goldsborough](https://github.com/goldsborough).

An example of writing a C++ extension for PyTorch. See
[here](http://pytorch.org/tutorials/advanced/cpp_extension.html) for the accompanying tutorial.

There are a few "sights" you can metaphorically visit in this repository:

- Inspect the C++ and CUDA extensions in the `cpp/` and `cuda/` folders,
- Build C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and executing `python setup.py install`,
- JIT-compile C++ and/or CUDA extensions by going into the `cpp/` or `cuda/` folder and calling `python jit.py`, which will JIT-compile the extension and load it,
- Benchmark Python vs. C++ vs. CUDA by running `python benchmark.py {py, cpp, cuda} [--cuda]`,
- Run gradient checks on the code by running `python grad_check.py {py, cpp, cuda} [--cuda]`.
- Run output checks on the code by running `python check.py {forward, backward} [--cuda]`.

## Mapping a 4D Torch Tensor to GPU Threads

- Example 1: AdaptiveMaxPooling2d, copied from [here](https://raw.githubusercontent.com/pytorch/pytorch/master/aten/src/ATen/native/cuda/AdaptiveMaxPooling2d.cu).
- Example 2: AveragePool2d, copied from [here](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/AveragePool2d.cu).
