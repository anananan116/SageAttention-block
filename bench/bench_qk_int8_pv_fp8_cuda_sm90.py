import torch
from flash_attn.utils.benchmark import benchmark_forward
import sageattention._qattn_sm90 as qattn
import argparse

parser = argparse.ArgumentParser(description='Benchmark QK Int8 PV FP8 SM90')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=5, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=128, help='Head dimension')
parser.add_argument('--pv_accum_dtype', type=str, default='fp32+fp32', choices=['fp32', 'fp32+fp32'])
parser.add_argument('--quant_gran', type=str, default='per_thread', choices=['per_warp', 'per_thread'], help='Quantization granularity')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

assert args.pv_accum_dtype == 'fp32+fp32', "pure fp32 accumulator is not supported for now"

print(f"CUDA QK Int8 PV FP8 SM90 Benchmark")
print(f"batch: {batch}, head: {head}, headdim: {headdim}")

WARP_Q = 32
WARP_K = 64

kernel = qattn.qk_int8_sv_f8_accum_f32_attn_inst_buf

_qk_quant_gran = 3 if args.quant_gran == 'per_thread' else 2
block_size = 16384
print(f"Causal")
for seq_len in {17160, 32760}:
    seq_len = ((seq_len + 127) // 128) * 128
    maskmode = 1 if seq_len == 17160 else 2
    flops = 4 * head * batch * headdim * seq_len * seq_len / 2

    q = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
    o = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")

    v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4 * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128 * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, head, headdim, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(100): kernel(q, k, v, o, q_scale, k_scale, 1, 1,# 0 for bidirectional, 1 for causal, 2 for block causal
                              _qk_quant_gran, sm_scale, 0, 0)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 1, 1,# 0 for bidirectional, 1 for causal, 2 for block causal
                                _qk_quant_gran, sm_scale, 0, 0, repeats=100, verbose=False, desc='Triton')
    print(f'seq_len:{seq_len} flops:{flops/time.mean*1e-12}')


print(f"Block Causal")
for seq_len in {17160, 32760}:
    if seq_len == 17160:
        flops = 4 * head * batch * headdim * seq_len * seq_len * (1-0.08264)
    else:
        flops = 4 * head * batch * headdim * seq_len * seq_len * 0.7278
    seq_len = ((seq_len + 127) // 128) * 128
    maskmode = 1 if seq_len == 17160 else 2

    q = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
    k = torch.randint(-95, 95, (batch, head, seq_len, headdim), dtype=torch.int8, device="cuda")
    o = torch.empty(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")

    v_scale = torch.randn(batch, head, headdim, dtype=torch.float, device="cuda")

    if args.quant_gran == 'per_warp':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128, dtype=torch.float, device="cuda")
    elif args.quant_gran == 'per_thread':
        q_scale = torch.randn(batch, head, seq_len // 64 * 4 * 8, dtype=torch.float, device="cuda")
        k_scale = torch.randn(batch, head, seq_len // 128 * 4, dtype=torch.float, device="cuda")

    v = torch.randn(batch, head, headdim, seq_len, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    sm_scale = 1 / (headdim ** 0.5)
    for i in range(100): kernel(q, k, v, o, q_scale, k_scale, 1, 2,# 0 for bidirectional, 1 for causal, 2 for block causal
                              _qk_quant_gran, sm_scale, 0, maskmode)
    torch.cuda.synchronize()
    _, time = benchmark_forward(kernel, q, k, v, o, q_scale, k_scale, 1, 2,# 0 for bidirectional, 1 for causal, 2 for block causal
                                _qk_quant_gran, sm_scale, 0, maskmode, repeats=100, verbose=False, desc='Triton')
    print(f'seq_len:{seq_len} flops:{flops/time.mean*1e-12}')