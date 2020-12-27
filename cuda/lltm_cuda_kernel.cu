/**
 * @file: lltm_cuda_kernel.cu
 * @brief:
 * @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
 * @version: 0.0.1
 * @creation date: 11-10-2019
 * @last modified: Fri 11 Oct 2019 05:27:09 PM EDT
 */

/*
 * Note that `setuptools` cannot handle files with the same name 
 * but different extensions, so if you use the setup.py method 
 * instead of the JIT method, you must give your CUDA file a 
 * different name than your C++ file (for the JIT method, lltm.cpp 
 * and lltm.cu would work fine).
 */

// Torch Extension
#include <torch/extension.h>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// C++
#include <vector>

namespace{

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

/* Here we see the headers I just described, 
	 as well as the fact that we are using CUDA-specific 
	 declarations like `__device__` and `__forceinline__` 
	 and functions like exp. Let’s continue with a few 
	 more helper functions that we’ll need:
*/

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(scalar_t(0.0), z) + fmin( scalar_t(0.0), alpha * (exp(z) - scalar_t(1.0)));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

#if 0
/* kernel functions： work directly on pointers with the right type;
 * Indeed, working directly with high level type agnostic tensors 
 * inside cuda kernels would be very inefficient.
 */
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
		const scalar_t* __restrict__ gates,
		const scalar_t* __restrict__ old_cell,
		scalar_t* __restrict__ new_h,
		scalar_t* __restrict__ new_cell,
		scalar_t* __restrict__ input_gate,
		scalar_t* __restrict__ output_gate,
		scalar_t* __restrict__ candidate_cell,
		size_t state_size){
	const int column = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = blockIdx.y * state_size + column;
	const int gates_row = blockIdx.y * (state_size * 3);
	if (column < state_size) {
		input_gate[index] = sigmoid(gates[gates_row + column]);
		output_gate[index] = sigmoid(gates[gates_row + state_size + column]);
		candidate_cell[index] = elu(gates[gates_row + 2 * state_size + column]);
		new_cell[index] =
			old_cell[index] + candidate_cell[index] * input_gate[index];
		new_h[index] = tanh(new_cell[index]) * output_gate[index];
	}
}
#endif



/* kernel functions：Using accessors!!!
 * Accessor objects have a relatively high level interface, with .size() and .stride() methods 
 * and multi-dimensional indexing. The .accessor<> interface is designed to access data 
 * efficiently on cpu tensor. The equivalent for cuda tensors are packed_accessor64<> 
 * and packed_accessor32<>, which produce Packed Accessors with either 64-bit or 32-bit integer indexing.
 * The fundamental difference with Accessor is that a Packed Accessor copies size and stride data 
 * inside of its structure instead of pointing to it. It allows us to pass it to a 
 * CUDA kernel function and use its interface inside it.
 */
template <typename scalar_t>
__global__ void lltm_cuda_forward_kernel(
		// ATen provides accessors that are created with a single dynamic check that a Tensor is the type and number of dimensions;
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,// assert gates is 3-dimensional and holds scalar_t type; 
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell)
{
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = elu(gates[n][2][c]);
    new_cell[n][c] = old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
    new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
  }
}


template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_old_cell,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> d_gates,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_h,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> grad_cell,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gate_weights) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_gates.size(2)){
    const auto d_output_gate = tanh(new_cell[n][c]) * grad_h[n][c];
    const auto d_tanh_new_cell = output_gate[n][c] * grad_h[n][c];
    const auto d_new_cell =
        d_tanh(new_cell[n][c]) * d_tanh_new_cell + grad_cell[n][c];


    d_old_cell[n][c] = d_new_cell;
    const auto d_candidate_cell = input_gate[n][c] * d_new_cell;
    const auto d_input_gate = candidate_cell[n][c] * d_new_cell;

    d_gates[n][0][c] =
        d_input_gate * d_sigmoid(gate_weights[n][0][c]);
    d_gates[n][1][c] =
        d_output_gate * d_sigmoid(gate_weights[n][1][c]);
    d_gates[n][2][c] =
        d_candidate_cell * d_elu(gate_weights[n][2][c]);
  }
}
} // namespace


std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {

  auto X = torch::cat({old_h, input}, /*dim=*/1);
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.reshape({batch_size, 3, state_size});
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);
  
  /*
	 * The main point of interest here is the AT_DISPATCH_FLOATING_TYPES macro 
	 * and the kernel launch (indicated by the <<<...>>>);
	 * The purpose of AT_DISPATCH_FLOATING_TYPES is to take care of the dispatch for us.
	 * It takes a type (gates.type() in our case), a name (for error messages) and a lambda function.
	 */

	/*
	 * Inside this lambda function, the type alias scalar_t is available and 
	 * is defined as the type that the tensor actually is at runtime 
	 * in that context. As such, if we have a template function (which 
	 * our CUDA kernel will be), we can instantiate it with this scalar_t alias, 
	 * and the correct function will be called. In this case, we also want 
	 * to retrieve the data pointers of the tensors as pointers of 
	 * that scalar_t type. If you wanted to dispatch over all types 
	 * and not just floating point types (Float and Double), 
	 * you can use AT_DISPATCH_ALL_TYPES.
	 */
  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}


std::vector<torch::Tensor> lltm_cuda_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gates,
    torch::Tensor weights) {
  auto d_old_cell = torch::zeros_like(new_cell);
  auto d_gates = torch::zeros_like(gates);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_forward_cuda", ([&] {
    lltm_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        d_gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        grad_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        grad_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>());
  }));

  auto d_gate_weights = d_gates.flatten(1, 2);
  auto d_weights = d_gate_weights.t().mm(X);
  auto d_bias = d_gate_weights.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gate_weights.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
