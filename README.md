# Assignment 13: Reversible transformers, a 20M-parameter LLM trained on 50M tokens

This repo trains a 20M-parameter decoder-only LLM on 50M tokens of FineWeb-Edu three times:

| # | run | residual stream | batch |
|---|---|---|---|
| 1 | `baseline` | standard (Euler) residual, normal backprop | **fixed**: largest power of two the baseline fits |
| 2 | `reversible` | reversible stepping rule (best of Hamiltonian / midpoint(a) / plain midpoint / leapfrog), activations rebuilt in backward | same fixed batch |
| 3 | `reversible_maxbatch` | same reversible model | **maximum** batch that fits |

For each run we report the final loss, throughput (tokens/s), peak GPU memory and other findings.

```
revllm/model.py      model, 3 reversible schemes (4 named variants) with exact inverses, RevStackFn (custom autograd)
revllm/data.py       FineWeb-Edu -> uint16 .bin tokenisation; deterministic batching
revllm/train.py      training loop, metrics, max-batch search
notebooks/00_data_prep.ipynb             tokenise 51M train + 1M val tokens (run once, stored on Drive)
notebooks/01_baseline.ipynb              run 1 (+ batch-size search)
notebooks/02_reversible.ipynb            gradient check, precision drift, memory table, variant sweep, run 2
notebooks/03_reversible_maxbatch.ipynb   memory vs depth, max-batch search, run 3
notebooks/04_results.ipynb               tables and plots -> results/
tests/test_reversible.py                 reversible grads == autograd grads (fp64); inverse round-trip; chunked CE;
                                         all params get grads under bf16 autocast (backward inside or outside)
```

---

## 1. Setup

| | |
|---|---|
| Model | decoder-only transformer, pre-RMSNorm, RoPE, SwiGLU MLP, tied embeddings |
| Size | d_model 320, 8 layers, 5 heads (head dim 64), d_ff 864, vocab 32 000 → **20.16M params** (10.24M embedding + 9.92M blocks) |
| Context | 512 tokens |
| Data | FineWeb-Edu `sample-10BT`, streamed; first 1M tokens → val, next 50M → train |
| Tokenizer | Llama-2 SentencePiece (via `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), 32k vocab, so tokens fit in uint16 |
| Token budget | one pass over the same shuffled order of 97 656 windows of 512 (50M tokens), fixed seed. A run at batch B trains on the first ⌊97 656 / B⌋·B windows, so every run sees a prefix of the same order and at most B − 1 windows are dropped; each run records its exact `tokens_trained` |
| Optimiser | AdamW (β 0.9/0.95, wd 0.1 on matrices), grad clip 1.0, linear warm-up then cosine decay to 10% of peak |
| LR | 1e-3 at the fixed batch; run 3 uses 1e-3·√(B_max/B_fixed), capped at 3e-3, with 10% warm-up |
| Precision | autocast bf16 (A100/L4) or fp16 with GradScaler (T4). The residual streams stay in fp32 |
| Loss | chunked cross-entropy (4096 rows per chunk, checkpointed), so the (B·T × 32k) fp32 logits never exist in full. All runs use it, so memory is dominated by the transformer body, which is the part reversibility changes |
| Metrics | **final train loss** = mean of the last 5% of logged steps; **final val loss** = 409k held-out tokens; **tokens/s** = train-step time only, excluding evals and the first logging window; **peak memory** = `torch.cuda.max_memory_allocated` (also reported: `max_memory_reserved`) |

No `torch.compile` in any run. The reversible autograd function calls `autograd.backward` inside its own backward, which compile does not trace, and leaving compile off everywhere keeps the throughput comparison fair.

---

## 2. Reversibility

A residual network is the explicit **Euler** discretisation of an ODE, `p_{j+1} = p_j + f_j(p_j)`, where `f_j` is block j's increment (attention followed by the MLP). **Euler is the non-reversible baseline, not one of the reversible variants.** Recovering `p_j` from `p_{j+1}` would mean solving `p = p_{j+1} − f(p)`, which has no closed form, so backprop has to store every `p_j`.

The reversible schemes carry **two** state tensors. Each step then has an *explicit* algebraic inverse, so backward can rebuild layer j's input from layer j+1's output. Only the final state is stored, and activation memory becomes **O(1) in depth** instead of O(L). Every hyper-parameter below is set explicitly (`revllm/model.py: VARIANTS`), because library defaults differ.

| key | scheme | forward step | exact inverse | output |
|---|---|---|---|---|
| `hamiltonian` | **Hamiltonian, a = b = 1** (identical to RevNet / Reformer additive coupling) | p' = p + Attn(q); q' = q + MLP(p') | q = q' − MLP(p'); p = p' − Attn(q) | (p+q)/2 |
| `midpoint_a0.5_h0.25` | **midpoint(a)**, the lesson's recipe (step 0.25, blend 0.5) | p_{j+1} = a·p_{j−1} + (1−a)·p_j + h·f_j(p_j) | p_{j−1} = (p_{j+1} − (1−a)·p_j − h·f_j(p_j)) / a | p_L |
| `midpoint_plain_h0.25` | plain midpoint, the paper's eq. (2.4) at h = 0.25 (coded as midpoint(a) with a = 1, h = 0.5) | p_{j+1} = p_{j−1} + 2h·f_j(p_j) | p_{j−1} = p_{j+1} − 2h·f_j(p_j) | (p_{L−1}+p_L)/2 |
| `leapfrog_h1` | velocity Verlet (second order) | v' = v + h·f(x); x' = x + h·v' | x = x' − h·v'; v = v' − h·f(x) | x_L |

All models have **identical parameter counts (20.16M)**. Only the wiring of the residual stream differs. The states are initialised to (p₋₁, p₀) = (e, e), or to (x, v) = (e, 0) for leapfrog, where e is the token embedding.

The two midpoint rows are two recipes, each at h = 0.25 in its own equation from the paper, not an isolation of a. They differ in three settings at once: the blend a (0.5 vs 1), the increment coefficient (h = 0.25 vs 2h = 0.5), and the readout (p_L vs the average of the last two states). A difference in their loss cannot be attributed to a alone.

Why midpoint(a) and not plain midpoint:
- **Stability.** On the test problem f(p) = λp, the recurrence p_{j+1} = a·p_{j−1} + (1−a)·p_j + c·λ·p_j (c = the increment coefficient) has roots r² − (1 − a + cλ)·r − a = 0. At a = 1 the roots at cλ = 0 are +1 and a parasitic −1. For a real **negative** eigenvalue, a direction the block damps, the parasitic root grows even though the true solution decays: at cλ = −0.5, |r| = 1.28 and 0.78. With a = 0.5 the parasitic root sits at −a, and both roots have |r| = 0.71 at cλ = −0.5. For a real positive eigenvalue the large root is the principal one, i.e. growth the underlying ODE has too (Euler also gives 1 + cλ = 1.5). The paper states midpoint's weakness as sensitivity to real positive eigenvalues; on this test problem the spurious growth is on the negative side.
- **Cost.** The paper's analysis requires |a| = 1 for stability in both directions. At a = 0.5 the forward pass is damped, and the inverse divides by a, so reconstruction error can grow by up to 1/a = 2 per layer, 2⁸ = 256× over 8 layers in the worst case. The measured penalty (below) is 4.5×, far under that bound.

### Backward pass: one extra block evaluation per layer

`RevStackFn` runs the forward pass with no graph and stores only the final (p, q). In backward, for each layer from the top down, `Stepper.backward_step` evaluates the block **once** with grad. That single evaluation provides both the reconstruction (from its detached output) and the VJP. For midpoint(a):

```python
y = f(p_j)                                           # with grad, evaluated once
p_prev = (p_next - (1-a)*p_j - h*y.detach()) / a     # reconstruction reuses it
# dL/dp_prev = a*g_next ;  dL/dp_j = g_j + (1-a)*g_next + vjp(h*y, g_next)
```

The Hamiltonian variant evaluates MLP(p') and then Attn(q) once each in the same way. Cost per layer is 1 forward + 1 re-evaluation + 1 backward ≈ **4 forward-equivalents vs 3 for the baseline** (about 33% more compute). That is the lesson's "one extra block evaluation". Checked with a forward hook on every attention and MLP module: one train step makes **4L** such calls for every reversible variant against **2L** for Euler, i.e. each block is evaluated twice instead of once. A local Apple-MPS run measured throughput at **0.84–0.86× the baseline** at the same batch; that figure is not reproduced by this repo's notebooks, and §3 reports the T4 number. Autocast state is captured in forward and restored in backward, so the re-evaluation runs in the same precision.

### Correctness and drift checks

Pre-run numbers, measured on CPU (PyTorch 2.5.1, 1 thread, bf16 autocast, random init, seed 0, full 20M model at B = 4 for the bf16 rows). Notebook 02 re-measures every row on the GPU, and those numbers replace these in §3.

| check | hamiltonian | midpoint(a=0.5, h=0.25) | plain midpoint | leapfrog |
|---|---|---|---|---|
| fp64 reversible vs autograd, max \|Δgrad\| (notebook-02 config) | 1.4e-17 | 1.0e-17 | 1.4e-17 | 5.6e-17 |
| bf16, 8-layer reconstruction rel. error | 1.7e-3 | **9.4e-3** | 2.1e-3 | 2.7e-2 |
| bf16, gradient rel. error vs autograd | 0.4% | 0.3% | 0.3% | 2.1% |

An earlier local Apple-MPS run reported gradient errors of 1.1 / 4.7 / 1.0 / 6.3%. They did not reproduce here, so they are withdrawn. In particular, midpoint(a) does **not** cost gradient accuracy at init: its penalty shows up only in reconstruction.

Activations saved for backward, B = 4, T = 512 (autograd saved-tensor hooks, each storage counted once, bf16 autocast, CPU). Notebook 02 has the same measurement on the GPU, including L = 32:

| layers | Euler | any reversible variant |
|---|---|---|
| 8 | 345 MB | 49–52 MB (6.6–7.0× less) |
| 16 | 643 MB | 49–52 MB (12.4–13.1×) |
| 32 | (notebook 02) | (notebook 02) |

Three things stand out. First, **depth drops out** of the reversible column. Second, midpoint(a)'s division by a costs 4.5× more reconstruction error than plain midpoint (9.4e-3 vs 2.1e-3), far below the 256× worst case. Third, that reconstruction penalty does not carry into the gradient at init. The T4 runs use fp16, which has 3 more mantissa bits than bf16, so drift there should be about 8× smaller.

**A bug these checks caught.** Autocast caches each weight's low-precision cast for the whole autocast region. `RevStackFn.forward` runs with grad disabled, so its cached casts carry no graph. When `backward()` was called *inside* the autocast region, the backward re-evaluation reused those casts, and 16 of 26 parameters (every block weight matrix) silently received no gradient. `train_step` calls backward outside autocast, so training was unaffected; notebook 02's drift cell calls it inside, and would have crashed. The fix is `cache_enabled=False` for the autocast context in the reversible backward (`revllm/model.py: _amp_state`), guarded by `tests/test_reversible.py::test_autocast_grads_present_and_close`. That test fails on the unfixed code (4 of 8 cases) and passes after the fix.

---

## 3. Results

> Filled in from `results/RESULTS.md`, which `notebooks/04_results.ipynb` generates.
> GPU: **<fill in: e.g. Tesla T4 16 GB, fp16>**

### Batch sizes

| | batch (sequences × 512) | tokens / step | optimizer steps for 50M tokens |
|---|---|---|---|
| baseline max | <fill> | | |
| **fixed** (runs 1 & 2) | <fill> | | |
| reversible max (run 3) | <fill> | | |
| reversible trained (run 3; = max unless the OOM fallback fired) | <fill> | | |

Prediction made before running, extrapolated from the saved-bytes numbers (about 86 MB per sequence for the baseline vs about 12.5 MB reversible, from the CPU table in §2); these are estimates, not measurements. On a 16 GB T4, baseline max ≈ 150 → fixed 128 → ≈ 760 steps. Reversible max ≈ 800–1000+ → only ≈ 100 steps for the same 50M tokens.

### Main table

| run | residual | batch | optimizer steps | final train loss | final val loss | tokens/s | peak mem alloc (GB) | peak mem reserved (GB) | wall time (min) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | euler | | | | | | | | |
| reversible | <variant> | | | | | | | | |
| reversible_maxbatch | <variant> | | | | | | | | |

### Which reversible variant worked (5M-token sweep at the fixed batch)

| variant | val loss @ 5M | tokens/s | peak mem (GB) | recon rel. err | grad rel. err |
|---|---|---|---|---|---|
| hamiltonian (a=b=1) | | | | | |
| midpoint(a=0.5, h=0.25) | | | | | |
| plain midpoint (h=0.25) | | | | | |
| leapfrog (h=1) | | | | | |

**Variant used for runs 2 and 3: `<fill>`**, because <fill: lowest val loss / stable / drift acceptable>.

### Memory vs depth (batch 16)

| layers | Euler peak (GB) | <variant> peak (GB) |
|---|---|---|
| 4 | | |
| 8 | | |
| 16 | | |
| 32 | | |

### Plots

![loss curves: vs tokens and vs optimizer steps](results/loss_curves.png)
![speed and memory](results/speed_memory.png)
![memory vs batch and depth](results/memory_vs_batch_depth.png)

---

## 4. Findings

<!-- Replace the bracketed parts with your numbers; keep the ones your data supports. -->

1. **Memory.** At the fixed batch, reversible backprop cut peak memory from <X> GB to <Y> GB. What remains is weights, AdamW state, the embedding output, the final two-tensor state, one layer's live graph during backward, and the CE chunk. The depth sweep shows Euler growing about linearly in L while the reversible model stays flat (<numbers>), which is the lesson's central claim. At only 8 layers the saving (about 6× in activations) understates what happens in deep models, where the paper reports about 10×.
2. **Throughput.** At the same batch the reversible model runs at <r>× the baseline's tokens/s. That matches the one-extra-block-evaluation cost (≈4/3 compute, 30–50% overhead).
3. **Max batch.** The reversible model fits <B_max> sequences against the baseline's <B_base> (<k>× larger), and tokens/s <rises/falls> to <…>. This batch increase is where reversibility's throughput gains come from.
4. **Loss at a fixed token budget.** Run 2 lands within <Δ> of the baseline: same parameter count, and reversible gradients equal autograd's up to rounding. Run 3 ends clearly **worse** (<Δ>). It sees the same 50M tokens, but in only <N> optimizer steps against <M>. The loss-vs-steps panel shows it on the same per-step curve, just stopped early. Past the critical batch size, extra batch buys memory headroom and hardware utilisation, not loss per token. At a fixed budget of steps, or of wall-clock time on a bigger model, the picture reverses.
5. **Variants.** <e.g. midpoint(a=0.5, h=0.25) trained best / most stable; plain midpoint <diverged | was worse>, consistent with its undamped parasitic mode; Hamiltonian <…>; leapfrog has the largest drift because its state grows quickly (velocity accumulates every increment).> The reconstruction error of the trained <variant> in <dtype> was <value>.
6. **Practical notes.**
   - The dropout rate must be 0 (or the RNG must be replayed), or the re-evaluation will not match the forward.
   - `torch.compile` does not trace the custom backward.
   - A custom recompute path must not reuse autocast's cast cache: the forward's no-grad casts will be reused by the backward re-evaluation if backward runs inside the same autocast region, dropping every block weight gradient without an error (see §2).
   - Chunked cross-entropy matters as much as reversibility: a full fp32 logits tensor at B = 64 is about 4 GB on its own.

---

## 5. Reproduce

1. Push this repo to GitHub as `gaurkhare/gaurav-eagv5-s13` (the `REPO_URL` in each notebook's first cell); change it there if the repo lives elsewhere.
2. In Colab, select a GPU runtime and run the notebooks **in order**: `00 → 01 → 02 → 03 → 04`. Data and results persist in `MyDrive/era_a13/`, so each notebook can run in a fresh session.
3. The last cell of `04` copies `results/*` into the repo checkout. Download it, or commit it from Colab, together with the executed notebooks.

Locally: `pip install -r requirements.txt`, then `pytest tests/`. The notebooks also run from `notebooks/` on CUDA, MPS or CPU. MPS reports driver memory, not true peak allocation, so memory numbers are only meaningful on CUDA.
