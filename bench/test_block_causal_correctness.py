import torch
import torch.nn.functional as F
import sageattention
import numpy as np

def create_block_causal_mask(seq_len, block_ends, device='cuda', dtype=torch.float32):
    """Create a block-wise causal attention mask for verification."""
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    block_starts = [0] + block_ends[:-1]
    
    for i, (start, end) in enumerate(zip(block_starts, block_ends)):
        # Within block i: bidirectional attention
        mask[start:end, start:end] = 1.0
        
        # Block i can attend to all previous blocks
        if i > 0:
            mask[start:end, :start] = 1.0
    
    return mask

def manual_block_causal_attention(q, k, v, block_ends, sm_scale):
    """Manual implementation of block-wise causal attention for verification."""
    batch, num_heads, seq_len, head_dim = q.shape
    
    # Compute QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    
    # Create and apply mask
    mask = create_block_causal_mask(seq_len, block_ends, device=q.device, dtype=scores.dtype)
    scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights

def visualize_attention_pattern(seq_len, block_ends):
    """Visualize the block-wise causal attention pattern."""
    mask = create_block_causal_mask(seq_len, block_ends, device='cpu', dtype=torch.float32)
    
    print("\nBlock-wise Causal Attention Pattern:")
    print(f"Sequence length: {seq_len}, Block ends: {block_ends}")
    print("\nMask (1 = can attend, 0 = masked):")
    
    # Convert to numpy for better printing
    mask_np = mask.cpu().numpy()
    
    # Print with block boundaries marked
    block_starts = [0] + block_ends[:-1]
    
    # Print column headers with block boundaries
    print("     ", end="")
    for j in range(seq_len):
        if j in block_starts[1:]:
            print("|", end="")
        else:
            print(" ", end="")
    print()
    
    # Print the mask
    for i in range(seq_len):
        if i in block_starts[1:]:
            print("----+" + "-" * seq_len)
        
        print(f"{i:3d}: ", end="")
        for j in range(seq_len):
            if j in block_starts[1:]:
                print("|", end="")
            print("■" if mask_np[i, j] == 1 else "·", end="")
        print()

def test_block_causal_correctness():
    print("Testing Block-wise Causal Attention Correctness")
    print("="*60)
    
    # Test configurations
    test_cases = [
        # (seq_len, block_ends, description)
        (8, [4, 8], "Two equal blocks"),
        (10, [3, 7, 10], "Three unequal blocks"),
        (16, [4, 8, 12, 16], "Four equal blocks"),
        (15, [5, 10, 15], "Three blocks of size 5"),
    ]
    
    for seq_len, block_ends, description in test_cases:
        print(f"\n\nTest Case: {description}")
        print(f"Sequence length: {seq_len}, Block ends: {block_ends}")
        
        # Visualize the attention pattern
        visualize_attention_pattern(seq_len, block_ends)
        
        # Create small test tensors
        batch = 2
        num_heads = 4
        head_dim = 64
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
        sm_scale = 1.0 / (head_dim ** 0.5)
        
        # Get reference output from manual implementation
        ref_output, ref_attn_weights = manual_block_causal_attention(q, k, v, block_ends, sm_scale)
        
        # Get SageAttention output
        sage_output = sageattention.sageattn(q, k, v, is_causal="block_causal", 
                                           block_ends=block_ends, sm_scale=sm_scale)
        
        # Compare outputs
        abs_error = torch.abs(sage_output - ref_output)
        rel_error = abs_error / (torch.abs(ref_output) + 1e-6)
        
        print(f"\nAccuracy Metrics:")
        print(f"  Max absolute error: {abs_error.max().item():.6f}")
        print(f"  Mean absolute error: {abs_error.mean().item():.6f}")
        print(f"  Max relative error: {rel_error.max().item():.6f}")
        print(f"  Mean relative error: {rel_error.mean().item():.6f}")
        
        # Check if errors are within acceptable bounds
        if abs_error.max().item() < 0.01 and rel_error.max().item() < 0.05:
            print("  ✓ PASSED: Errors within acceptable bounds")
        else:
            print("  ✗ FAILED: Errors exceed acceptable bounds")
    
    # Test edge cases
    print("\n\n" + "="*60)
    print("Testing Edge Cases:")
    
    # Edge case 1: Single block (should behave like bidirectional attention)
    print("\n1. Single block (full bidirectional attention):")
    seq_len = 16
    block_ends = [16]
    
    q = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda")
    
    sage_output = sageattention.sageattn(q, k, v, is_causal="block_causal", 
                                       block_ends=block_ends, sm_scale=1.0/8)
    ref_output = F.scaled_dot_product_attention(q, k, v, scale=1.0/8)
    
    error = torch.abs(sage_output - ref_output).max().item()
    print(f"  Max error vs full attention: {error:.6f}")
    print(f"  {'✓ PASSED' if error < 0.01 else '✗ FAILED'}")
    
    # Edge case 2: Many small blocks (each position is its own block)
    print("\n2. Many small blocks (essentially causal):")
    seq_len = 8
    block_ends = list(range(1, seq_len + 1))  # [1, 2, 3, 4, 5, 6, 7, 8]
    
    q = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda") 
    v = torch.randn(1, 1, seq_len, 64, dtype=torch.float16, device="cuda")
    
    sage_output = sageattention.sageattn(q, k, v, is_causal="block_causal", 
                                       block_ends=block_ends, sm_scale=1.0/8)
    ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0/8)
    
    error = torch.abs(sage_output - ref_output).max().item()
    print(f"  Max error vs causal attention: {error:.6f}")
    print(f"  {'✓ PASSED' if error < 0.01 else '✗ FAILED'}")

if __name__ == "__main__":
    test_block_causal_correctness() 