import triton
import triton.language as tl
import torch

@triton.jit
def full_2d_reduction_kernel(input_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute program ID for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros([1], dtype=tl.float32)
    
    # Compute pointer offsets for 2D tensor
    input_offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Load input block
    x = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # Sum all elements in the block
    block_sum = tl.sum(x, axis=(0, 1))
    
    # Atomically add to the output scalar
    tl.atomic_add(output_ptr, block_sum)

# Example launcher
def full_2d_reduction(input_tensor):
    M, N = input_tensor.shape
    output = torch.zeros(1, dtype=torch.float32, device=input_tensor.device)
    BLOCK_M = 32
    BLOCK_N = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), 1)
    full_2d_reduction_kernel[grid](input_tensor, output, M, N, BLOCK_M, BLOCK_N)
    return output[0]

# Test the kernel
input = torch.randn(128, 256, device='cuda', dtype=torch.float32)
output = full_2d_reduction(input)
torch_sum = torch.sum(input, dim=(0, 1))
assert torch.allclose(output, torch_sum, atol=1e-5), "Results do not match!"
print("Results match torch.sum!")