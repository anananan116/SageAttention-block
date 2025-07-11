import torch
import torch.nn.functional as F
from flash_attn.utils.benchmark import benchmark_forward
import sageattention
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.quant import per_channel_fp8
import argparse
import time
import numpy as np
import sageattention._qattn_sm90 as qattn

# Check if flex_attention is available (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = False
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("Warning: flex_attention not available. Please upgrade to PyTorch 2.5+ for flex attention benchmarks.")

def create_block_causal_mask(seq_len, block_ends, device='cuda', dtype=torch.float32):
    """Create a block-wise causal attention mask.
    
    Within each block: bidirectional attention (all 1s)
    Between blocks: causal (only attend to earlier blocks)
    """
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    block_starts = [0] + block_ends[:-1]
    
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        # Within block i: bidirectional attention
        mask[start:end, start:end] = 1.0
        
        # Block i can attend to all previous blocks
        if i > 0:
            mask[start:end, :start] = 1.0
    
    # Convert to attention mask format (0 for attend, -inf for mask)
    attention_mask = torch.where(mask == 1.0, 0.0, float('-inf'))
    # Ensure the attention mask has the correct dtype
    attention_mask = attention_mask.to(dtype=dtype)
    return attention_mask

def pytorch_block_causal_attention(q, k, v, block_ends, sm_scale):
    """PyTorch implementation of block-wise causal attention using scaled_dot_product_attention."""
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Create block causal mask
    mask = create_block_causal_mask(seq_len, block_ends, device=q.device, dtype=q.dtype)
    
    # Apply scaled_dot_product_attention with custom mask
    # Need to expand mask for batch and heads
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=sm_scale)
    return output

def create_flex_block_causal_mask(seq_len, block_ends, device='cuda'):
    """Create a block causal mask for flex attention."""
    if not FLEX_ATTENTION_AVAILABLE:
        return None
        
    block_starts = [0] + block_ends[:-1]
    
    # Pre-compute block membership for all positions
    # This avoids data-dependent control flow
    block_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        block_ids[start:end] = i
    
    def block_causal_mask_fn(b, h, q_idx, kv_idx):
        # Get block IDs for q and kv positions
        q_block = block_ids[q_idx]
        kv_block = block_ids[kv_idx]
        
        # Within same block: bidirectional attention
        # Can attend to earlier blocks or same block
        return kv_block <= q_block
    
    # Create block mask
    
    block_mask = create_block_mask(
        block_causal_mask_fn,
        B=1,  # Will be broadcasted
        H=1,  # Will be broadcasted  
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )
    
    return block_mask

def flex_block_causal_attention(q, k, v, block_ends, sm_scale):
    """Flex attention implementation of block-wise causal attention."""
    if not FLEX_ATTENTION_AVAILABLE:
        return None
        
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Create block mask
    block_mask = create_flex_block_causal_mask(seq_len, block_ends, device=q.device)
    
    # Apply flex attention (without compile for now, as it might cause issues)
    output = flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale)
    return output
@torch.no_grad()
def benchmark_block_causal():
    parser = argparse.ArgumentParser(description='Benchmark Block-wise Causal Attention')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
    parser.add_argument('--block_size', type=int, default=2048, help='Size of each attention block')
    parser.add_argument('--quant_gran', type=str, default='per_thread', choices=['per_warp', 'per_thread'], help='Quantization granularity')
    args = parser.parse_args()

    batch = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    block_size = args.block_size
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Flex Attention available: {FLEX_ATTENTION_AVAILABLE}")
    print()
    print(f"Block-wise Causal Attention Benchmark")
    print(f"batch: {batch}, num_heads: {num_heads}, head_dim: {head_dim}, block_size: {block_size}")
    print(f"quantization granularity: {args.quant_gran}")
    print("="*80)
    
    # Test different sequence lengths
    seq_lens = [2048, 4096, 8192, 16384]
    kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf
    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")
        
        # Calculate block boundaries
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ends = [min(i * block_size, seq_len) for i in range(1, num_blocks + 1)]
        print(f"Number of blocks: {num_blocks}, block_ends: {block_ends}")
        
        # Generate random tensors
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Check if flex attention works for this configuration
        flex_works = FLEX_ATTENTION_AVAILABLE
        if flex_works:
            _ = flex_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        # Warmup
        for _ in range(5):
            _ = pytorch_block_causal_attention(q, k, v, block_ends, sm_scale)
            _ = sageattention.sageattn(q, k, v, is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale)
            if flex_works:
                _ = flex_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        _, pytorch_time = benchmark_forward(
            pytorch_block_causal_attention, q, k, v, block_ends, sm_scale,
            repeats=100, verbose=False, desc='PyTorch'
        )
        ref_output = pytorch_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        # Benchmark Flex Attention
        if flex_works:
            _, flex_time = benchmark_forward(
                flex_block_causal_attention, q, k, v, block_ends, sm_scale,
                repeats=100, verbose=False, desc='FlexAttention'
            )
            flex_output = flex_block_causal_attention(q, k, v, block_ends, sm_scale)
        else:
            flex_time = None
            flex_output = None
        
        # Benchmark SageAttention
        _, sage_time = benchmark_forward(
            sageattention.sageattn, q, k, v, 
            is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale,
            repeats=100, verbose=False, desc='SageAttention'
        )
        sage_output = sageattention.sageattn(q, k, v, is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale)
        
        # Calculate speedup
        speedup_vs_pytorch = pytorch_time.mean / sage_time.mean
        if flex_works:
            speedup_vs_flex = flex_time.mean / sage_time.mean
        
        # Calculate accuracy (relative error)
        ref_output_for_accuracy = pytorch_block_causal_attention(q, k, v, block_ends, sm_scale)
        sage_output_for_accuracy = sageattention.sageattn(q, k, v, is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale)
        
        abs_error = torch.abs(sage_output_for_accuracy - ref_output_for_accuracy)
        rel_error = abs_error / (torch.abs(ref_output_for_accuracy) + 1e-6)
        max_abs_error = abs_error.max().item()
        mean_abs_error = abs_error.mean().item()
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        # Calculate FLOPS
        # For block-wise causal, the number of operations depends on block structure
        # Within each block: bidirectional attention
        # Between blocks: causal attention
        total_ops = 0
        block_starts = [0] + block_ends[:-1]
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            block_len = end - start
            # Operations within this block (bidirectional)
            total_ops += block_len * block_len
            # Operations from this block to previous blocks
            if i > 0:
                total_ops += block_len * start
        
        flops = 4 * num_heads * batch * head_dim * total_ops
        pytorch_tflops = flops / pytorch_time.mean / 1e12
        sage_tflops = flops / sage_time.mean / 1e12
        if flex_works:
            flex_tflops = flops / flex_time.mean / 1e12
        
        print(f"\nPerformance Results:")
        print(f"  PyTorch time: {pytorch_time.mean*1000:.2f} ms ({pytorch_tflops:.2f} TFLOPS)")
        if flex_works:
            print(f"  FlexAttention time: {flex_time.mean*1000:.2f} ms ({flex_tflops:.2f} TFLOPS)")
        print(f"  SageAttention time: {sage_time.mean*1000:.2f} ms ({sage_tflops:.2f} TFLOPS)")
        print(f"  Speedup vs PyTorch: {speedup_vs_pytorch:.2f}x")
        if flex_works:
            print(f"  Speedup vs FlexAttention: {speedup_vs_flex:.2f}x")
        
        print(f"\nAccuracy Results (vs PyTorch):")
        print(f"  Max absolute error: {max_abs_error:.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.6f}")
        print(f"  Max relative error: {max_rel_error:.6f}")
        print(f"  Mean relative error: {mean_rel_error:.6f}")
        
        # Also check flex attention accuracy if available
        if flex_works:
            flex_output_for_accuracy = flex_block_causal_attention(q, k, v, block_ends, sm_scale)
            flex_abs_error = torch.abs(sage_output_for_accuracy - flex_output_for_accuracy)
            flex_rel_error = flex_abs_error / (torch.abs(flex_output_for_accuracy) + 1e-6)
            print(f"\nAccuracy Results (vs FlexAttention):")
            print(f"  Max absolute error: {flex_abs_error.max().item():.6f}")
            print(f"  Mean absolute error: {flex_abs_error.mean().item():.6f}")
            print(f"  Max relative error: {flex_rel_error.max().item():.6f}")
            print(f"  Mean relative error: {flex_rel_error.mean().item():.6f}")
        
        print("-"*80)
    
    # Additional test: varying block sizes
    print("\n" + "="*80)
    print("Testing different block sizes (seq_len=8192):")
    seq_len = 8192
    block_sizes = [512, 1024, 2048, 4096]
    
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    
    for block_size in block_sizes:
        print(f"\nBlock size: {block_size}")
        
        # Calculate block boundaries
        num_blocks = (seq_len + block_size - 1) // block_size
        block_ends = [min(i * block_size, seq_len) for i in range(1, num_blocks + 1)]
        print(f"Number of blocks: {num_blocks}")
        
        # Check if flex attention works for this configuration
        flex_works = FLEX_ATTENTION_AVAILABLE
        if flex_works:
            flex_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        # Warmup
        for _ in range(5):
            _ = pytorch_block_causal_attention(q, k, v, block_ends, sm_scale)
            _ = sageattention.sageattn(q, k, v, is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale)
            if flex_works:
                _ = flex_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        torch.cuda.synchronize()
        
        # Benchmark
        _, pytorch_time = benchmark_forward(
            pytorch_block_causal_attention, q, k, v, block_ends, sm_scale,
            repeats=100, verbose=False, desc='PyTorch'
        )
        
        if flex_works:
            _, flex_time = benchmark_forward(
                flex_block_causal_attention, q, k, v, block_ends, sm_scale,
                repeats=100, verbose=False, desc='FlexAttention'
            )
        
        _, sage_time = benchmark_forward(
            sageattention.sageattn, q, k, v,
            is_causal="block_causal", block_ends=block_ends, sm_scale=sm_scale,
            repeats=100, verbose=False, desc='SageAttention'
        )
        
        speedup = pytorch_time.mean / sage_time.mean
        print(f"  PyTorch: {pytorch_time.mean*1000:.2f} ms, SageAttention: {sage_time.mean*1000:.2f} ms, Speedup: {speedup:.2f}x")
        if flex_works:
            flex_speedup = flex_time.mean / sage_time.mean
            print(f"  FlexAttention: {flex_time.mean*1000:.2f} ms, Speedup vs FlexAttention: {flex_speedup:.2f}x")

if __name__ == "__main__":
    benchmark_block_causal() 