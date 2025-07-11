import torch
import torch.nn.functional as F
from flash_attn.utils.benchmark import benchmark_forward
import sageattention
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.quant import per_channel_fp8
import argparse
import time
import numpy as np

def pytorch_causal_attention(q, k, v, sm_scale):
    """PyTorch implementation of causal attention using scaled_dot_product_attention."""
    # Use PyTorch's built-in causal attention
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
    return output

def benchmark_causal():
    parser = argparse.ArgumentParser(description='Benchmark Causal Attention')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--quant_gran', type=str, default='per_thread', choices=['per_warp', 'per_thread'], help='Quantization granularity')
    args = parser.parse_args()

    batch = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    
    print(f"Causal Attention Benchmark")
    print(f"batch: {batch}, num_heads: {num_heads}, head_dim: {head_dim}")
    print(f"quantization granularity: {args.quant_gran}")
    print("="*80)
    
    # Test different sequence lengths
    seq_lens = [2048, 4096, 8192, 16384]
    
    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")
        
        # Generate random tensors
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Warmup
        for _ in range(5):
            _ = pytorch_causal_attention(q, k, v, sm_scale)
            _ = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            ref_output = pytorch_causal_attention(q, k, v, sm_scale)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 50
        
        # Benchmark SageAttention
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            sage_output = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()
        sage_time = (time.time() - start_time) / 50
        
        # Calculate speedup
        speedup = pytorch_time / sage_time
        
        # Calculate accuracy (relative error)
        ref_output_for_accuracy = pytorch_causal_attention(q, k, v, sm_scale)
        sage_output_for_accuracy = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        
        abs_error = torch.abs(sage_output_for_accuracy - ref_output_for_accuracy)
        rel_error = abs_error / (torch.abs(ref_output_for_accuracy) + 1e-6)
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        # Calculate FLOPS for causal attention
        # Causal attention computes roughly half the operations of full attention
        flops = 4 * num_heads * batch * head_dim * seq_len * seq_len / 2
        pytorch_tflops = flops / pytorch_time / 1e12
        sage_tflops = flops / sage_time / 1e12
        
        print(f"\nPerformance Results:")
        print(f"  PyTorch time: {pytorch_time*1000:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
        print(f"  SageAttention time: {sage_time*1000:.2f} ms ({sage_tflops:.2f} TFLOPS)")
        print(f"  Speedup: {speedup:.2f}x")
        
        print(f"\nAccuracy Results:")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")
        print("-"*80)
    
    # Additional test: compare different head dimensions
    print("\n" + "="*80)
    print("Testing different head dimensions (seq_len=4096):")
    seq_len = 4096
    head_dims = [64, 128, 256]
    
    for head_dim in head_dims:
        print(f"\nHead dimension: {head_dim}")
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Warmup
        for _ in range(5):
            _ = pytorch_causal_attention(q, k, v, sm_scale)
            _ = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            ref_output = pytorch_causal_attention(q, k, v, sm_scale)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / 50
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(50):
            sage_output = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        torch.cuda.synchronize()
        sage_time = (time.time() - start_time) / 50
        
        speedup = pytorch_time / sage_time
        
        # Calculate accuracy
        ref_output_for_accuracy = pytorch_causal_attention(q, k, v, sm_scale)
        sage_output_for_accuracy = sageattention.sageattn(q, k, v, is_causal=True, sm_scale=sm_scale)
        
        abs_error = torch.abs(sage_output_for_accuracy - ref_output_for_accuracy)
        rel_error = abs_error / (torch.abs(ref_output_for_accuracy) + 1e-6)
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        print(f"  PyTorch: {pytorch_time*1000:.2f} ms, SageAttention: {sage_time*1000:.2f} ms, Speedup: {speedup:.2f}x")
        print(f"  Max abs error: {max_abs_error:.6f}, Mean abs error: {mean_abs_error:.6f}")
        print(f"  Max rel error: {max_rel_error:.6f}, Mean rel error: {mean_rel_error:.6f}")

if __name__ == "__main__":
    benchmark_causal()

