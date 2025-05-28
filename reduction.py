import triton
import triton.language as tl
import torch

"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------
DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def reduction_kernel(x_ptr,  # *Pointer* to first input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    pid_x = tl.program_id(0)  # Program ID for the first dimension.
    block_start = pid_x * BLOCK_SIZE  # Start of the block.
    offset = block_start + tl.arange(0, BLOCK_SIZE)  # Offset for each program.
    mask = offset < n_elements  # Mask to avoid out-of-bounds access.
    x = tl.load(x_ptr + offset, mask=mask)  # Load data from the input vector.
    tl.atomic_add(output_ptr, tl.sum(x))  # Atomic

# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:
def reduction(x: torch.Tensor):
    # We need to preallocate the output.
    output = torch.zeros([1], device=DEVICE, dtype=x.dtype)
    assert x.device == DEVICE and output.device == DEVICE
    n_elements = x.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    reduction_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


@triton.jit
def reduction_2d_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute program ID and offsets
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    output = tl.zeros([BLOCK_M], dtype=tl.float32)
    # Loop over N dimension in blocks
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        # Compute pointer offsets
        offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offs = offs_m[:, None] + offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
        acc = tl.max(x, axis=1)  # Reduce along axis 0 (sum rows)

    tl.atomic_add(output_ptr, tl.max(acc, axis=0))


# Example launcher
def reduction_2d(input_tensor):
    M, N = input_tensor.shape
    output = torch.zeros([1], dtype=torch.float32, device=input_tensor.device)
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), 1)
    reduction_2d_kernel[grid](input_tensor, output, M, N, BLOCK_M, BLOCK_N)
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:
x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
print(f'Triton result: {reduction(x).item()}')  # Check the sum using Triton
print(f'Torch result: {torch.sum(x,dim=0)}')  # Check the sum using PyTorch


# 2D example
input = torch.randn(128, 128, device='cuda', dtype=torch.float32)
print(f'Input: {input}')
triton_output = reduction_2d(input)
torch_output = torch.max(input)
print(f'Triton result: {triton_output.item()}')  # Check the sum using Triton
print(f'Torch result: {torch_output}')  # Check the sum using PyTorch



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['n'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='reduction-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(n, provider):
    x = torch.rand((n, n), device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sum(x, dim=[0, 1]), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: reduction(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='./')