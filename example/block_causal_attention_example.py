"""
Example of using Block-wise Causal Attention in SageAttention

Block-wise causal attention allows bidirectional attention within blocks
while maintaining causal relationships between blocks. This is useful for:
- Documents with natural boundaries (chapters, sections, paragraphs)
- Long sequences that can be divided into semantic chunks
- Structured data with hierarchical relationships
"""

import torch
import sageattention

def example_block_causal_attention():
    # Configuration
    batch_size = 2
    num_heads = 8
    seq_len = 12000  # Long sequence
    head_dim = 128
    
    # Define block boundaries
    # Example: 3 blocks representing document sections
    # Block 1: positions 0-3999 (introduction)
    # Block 2: positions 4000-7999 (main content)
    # Block 3: positions 8000-11999 (conclusion)
    block_ends = [4000, 8000, 12000]
    
    print("Block-wise Causal Attention Example")
    print("="*50)
    print(f"Sequence length: {seq_len}")
    print(f"Block boundaries: {block_ends}")
    print(f"Block sizes: {[block_ends[0]] + [block_ends[i] - block_ends[i-1] for i in range(1, len(block_ends))]}")
    
    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    
    # Apply block-wise causal attention
    output = sageattention.sageattn(
        q, k, v,
        is_causal="block_causal",
        block_ends=block_ends
    )
    
    print(f"\nOutput shape: {output.shape}")
    print("Block-wise causal attention applied successfully!")
    
    # Example with unequal block sizes
    print("\n" + "="*50)
    print("Example with unequal block sizes:")
    
    # Simulating a conversation or dialogue
    # Block 1: System prompt (small)
    # Block 2: User input (medium)
    # Block 3: Assistant response (large)
    seq_len_dialogue = 8192
    block_ends_dialogue = [512, 2048, 8192]  # 512, 1536, 6144 tokens respectively
    
    print(f"Sequence length: {seq_len_dialogue}")
    print(f"Block boundaries: {block_ends_dialogue}")
    print(f"Block sizes: {[block_ends_dialogue[0]] + [block_ends_dialogue[i] - block_ends_dialogue[i-1] for i in range(1, len(block_ends_dialogue))]}")
    
    q2 = torch.randn(1, 16, seq_len_dialogue, head_dim, 
                     dtype=torch.float16, device="cuda")
    k2 = torch.randn(1, 16, seq_len_dialogue, head_dim, 
                     dtype=torch.float16, device="cuda")
    v2 = torch.randn(1, 16, seq_len_dialogue, head_dim, 
                     dtype=torch.float16, device="cuda")
    
    output2 = sageattention.sageattn(
        q2, k2, v2,
        is_causal="block_causal",
        block_ends=block_ends_dialogue
    )
    
    print(f"\nOutput shape: {output2.shape}")
    
    # Demonstrate the attention pattern
    print("\n" + "="*50)
    print("Attention Pattern Explanation:")
    print("- Tokens within the same block can attend to each other (bidirectional)")
    print("- Tokens can attend to all tokens in earlier blocks (causal)")
    print("- Tokens cannot attend to tokens in later blocks")
    print("\nFor the dialogue example:")
    print("- System prompt (0-511): Can only see itself")
    print("- User input (512-2047): Can see system prompt and itself")
    print("- Assistant response (2048-8191): Can see everything before and itself")

def benchmark_vs_standard_attention():
    """Compare block-wise causal with standard causal attention"""
    print("\n" + "="*50)
    print("Comparing Block-wise vs Standard Causal Attention")
    
    batch_size = 1
    num_heads = 32
    seq_len = 4096
    head_dim = 128
    
    # Create tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    dtype=torch.float16, device="cuda")
    
    # Standard causal attention
    output_causal = sageattention.sageattn(q, k, v, is_causal=True)
    
    # Block-wise causal with small blocks (similar to causal)
    block_ends_small = list(range(64, seq_len + 1, 64))  # 64-token blocks
    output_block_small = sageattention.sageattn(
        q, k, v, is_causal="block_causal", block_ends=block_ends_small
    )
    
    # Block-wise causal with large blocks
    block_ends_large = [1024, 2048, 3072, 4096]  # 1024-token blocks
    output_block_large = sageattention.sageattn(
        q, k, v, is_causal="block_causal", block_ends=block_ends_large
    )
    
    print(f"\nStandard causal output shape: {output_causal.shape}")
    print(f"Block-wise (64-token blocks) output shape: {output_block_small.shape}")
    print(f"Block-wise (1024-token blocks) output shape: {output_block_large.shape}")
    
    # Compare differences
    diff_small = torch.abs(output_causal - output_block_small).mean().item()
    diff_large = torch.abs(output_causal - output_block_large).mean().item()
    
    print(f"\nMean absolute difference:")
    print(f"  Causal vs Block-wise (small blocks): {diff_small:.6f}")
    print(f"  Causal vs Block-wise (large blocks): {diff_large:.6f}")
    print("\nNote: Larger blocks allow more bidirectional attention,")
    print("resulting in bigger differences from standard causal attention.")

if __name__ == "__main__":
    example_block_causal_attention()
    benchmark_vs_standard_attention() 