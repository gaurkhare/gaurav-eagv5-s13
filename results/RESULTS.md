GPU: **Tesla T4**, precision: torch.bfloat16

Baseline max batch: 120, fixed batch: 64, reversible max batch: 456 (trained at 408); chosen variant: leapfrog_h1 {'residual': 'leapfrog', 'h': 1.0}

| run | residual | batch (seq × 512) | optimizer steps | final train loss | final val loss | tokens/s | peak mem alloc (GB) | peak mem reserved (GB) | wall time (min) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | euler | 64 | 1525 | 4.0896 | 4.1205 | 11,782 | 8.10 | 8.58 | 72.1 |
| reversible | leapfrog_h1 | 64 | 1525 | 4.0769 | 4.1055 | 8,275 | 7.14 | 8.23 | 102.2 |
| reversible_maxbatch | leapfrog_h1 | 408 | 239 | 5.4475 | 5.4190 | 9,420 | 12.64 | 14.42 | 92.5 |

- **reversible** vs baseline: throughput 0.70×, peak memory 0.88×, val loss -0.0150
- **reversible_maxbatch** vs baseline: throughput 0.80×, peak memory 1.56×, val loss +1.2985

### Reversible variant sweep (5M tokens, fixed batch)

| variant | val loss @5M tokens | tokens/s | peak mem (GB) | recon rel. err (init) | grad rel. err vs autograd (init) |
|---|---|---|---|---|---|
| hamiltonian | 6.3162 | 8,418 | 7.05 | 1.24e-03 | 0.37% |
| leapfrog_h1 | 6.0400 | 8,306 | 7.14 | 2.92e-02 | 2.98% |
| midpoint_a0.5_h0.25 | 6.3351 | 8,285 | 7.13 | 1.48e-02 | 0.36% |
| midpoint_plain_h0.25 | 6.1561 | 8,316 | 7.13 | 2.10e-03 | 0.34% |

Trained-model reconstruction error: {'variant': 'leapfrog_h1', 'recon_rel_err': 0.21365098655223846, 'dtype': 'torch.bfloat16'}

### Activations saved for backward (B = 4, T = 512)

| layers | euler (MB) | hamiltonian (MB) | midpoint_a0.5_h0.25 (MB) | midpoint_plain_h0.25 (MB) | leapfrog_h1 (MB) |
|---|---|---|---|---|---|
| 8 | 462 | 49 | 47 | 49 | 47 |
| 16 | 880 | 49 | 47 | 49 | 47 |
| 32 | 1716 | 49 | 47 | 49 | 47 |

### Memory at the fixed batch

| model | peak mem at fixed batch (GB) |
|---|---|
| euler | 8.11 |
| hamiltonian | 8.11 |
| hamiltonian (reversible backprop) | 2.20 |
| midpoint_a0.5_h0.25 | 8.11 |
| midpoint_a0.5_h0.25 (reversible backprop) | 2.18 |
| midpoint_plain_h0.25 | 8.11 |
| midpoint_plain_h0.25 (reversible backprop) | 2.14 |
| leapfrog_h1 | 8.11 |
| leapfrog_h1 (reversible backprop) | 2.26 |

### Peak memory vs depth (batch 16)

| layers | euler (GB) | leapfrog_h1 (GB) |
|---|---|---|
| 4 | 2.50 | 1.73 |
| 8 | 3.35 | 1.78 |
| 16 | 5.03 | 1.89 |
| 32 | 8.40 | 2.11 |
