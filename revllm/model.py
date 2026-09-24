"""~20M-parameter decoder-only LLM with optional reversible residual streams.

Residual variants (``cfg.residual``); f_j(x) is the usual pre-norm block increment attn + mlp:

  euler        p_{j+1} = p_j + f_j(p_j)                         standard transformer (NOT reversible:
                                                                recovering p_j needs solving p = p_{j+1} - f(p))
  hamiltonian  p' = p + Attn(q);  q' = q + MLP(p')              two-stream Hamiltonian scheme with a = b = 1
                                                                (= RevNet / Reformer additive coupling)
  midpoint     p_{j+1} = a p_{j-1} + (1-a) p_j + h f_j(p_j)     midpoint(a) method; default a = 0.5, h = 0.25.
                                                                a = 1 is plain midpoint p_{j-1} + 2h' f with h = 2h'
  leapfrog     v_{j+1} = v_j + h f_j(x_j);  x_{j+1} = x_j + h v_{j+1}   velocity Verlet, default h = 1

Every reversible variant carries a 2-tensor state (p, q) and has an exact algebraic inverse. With
``cfg.rev_backprop=True`` the stack stores only its final state. In backward, each layer evaluates
its block once with grad: that single evaluation both reconstructs the layer's input and supplies
the vector-Jacobian product, so the overhead is one extra block forward per layer.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    seq_len: int = 512
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 5
    d_ff: int = 864
    residual: str = "euler"      # euler | hamiltonian | midpoint | leapfrog
    rev_backprop: bool = True    # only for reversible variants: recompute instead of storing activations
    h: float | None = None       # step size (defaults: midpoint 0.25, leapfrog 1.0); set explicitly in runs
    a: float | None = None       # midpoint blend coefficient (default 0.5; a = 1 is plain midpoint)
    ce_chunk: int = 4096         # rows per chunk for the memory-light cross-entropy (0 = off)


REVERSIBLE = ("hamiltonian", "midpoint", "leapfrog")
DEFAULTS = {"midpoint": dict(a=0.5, h=0.25), "leapfrog": dict(h=1.0)}

# Named reversible variants compared in the sweep (all hyper-parameters set explicitly).
VARIANTS = {
    "hamiltonian": dict(residual="hamiltonian"),                        # a = b = 1 (RevNet coupling)
    "midpoint_a0.5_h0.25": dict(residual="midpoint", a=0.5, h=0.25),    # the lesson's recipe
    "midpoint_plain_h0.25": dict(residual="midpoint", a=1.0, h=0.5),    # p_{j-1} + 2*0.25*f(p_j): ablation
    "leapfrog_h1": dict(residual="leapfrog", h=1.0),
}


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------
def rope_cache(seq_len, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[: x.size(-2)].to(x.dtype), sin[: x.size(-2)].to(x.dtype)
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(self.norm(x)).view(B, T, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.RMSNorm(cfg.d_model)
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(self.norm(x)).chunk(2, dim=-1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def delta(self, x, cos, sin):
        """Residual increment D(x) of a standard pre-norm block, i.e. block(x) - x."""
        a = self.attn(x, cos, sin)
        return a + self.mlp(x + a)


# ---------------------------------------------------------------------------------------------
# reversible stepping rules: step(), exact inverse(), and a fused inverse + VJP for backward
# ---------------------------------------------------------------------------------------------
class Stepper:
    def __init__(self, kind, h=None, a=None):
        self.kind, self.h, self.a = kind, h, a

    def init(self, x):
        if self.kind == "leapfrog":
            return x, torch.zeros_like(x)          # (position, velocity)
        return x, x                                 # hamiltonian: (p, q); midpoint: (p_{j-1}, p_j)

    def step(self, blk, p, q, cos, sin):
        if self.kind == "hamiltonian":
            p = p + blk.attn(q, cos, sin)
            return p, q + blk.mlp(p)
        if self.kind == "midpoint":
            a, h = self.a, self.h
            return q, a * p + (1 - a) * q + h * blk.delta(q, cos, sin)
        v = q + self.h * blk.delta(p, cos, sin)     # leapfrog
        return p + self.h * v, v

    def inverse(self, blk, p, q, cos, sin):
        if self.kind == "hamiltonian":
            q = q - blk.mlp(p)
            return p - blk.attn(q, cos, sin), q
        if self.kind == "midpoint":
            a, h = self.a, self.h
            return (q - (1 - a) * p - h * blk.delta(p, cos, sin)) / a, p
        p = p - self.h * q
        return p, q - self.h * blk.delta(p, cos, sin)

    def backward_step(self, blk, p, q, gp, gq, cos, sin):
        """Given a layer's output state and its grads, return the input state and its grads.

        Each sub-block is evaluated exactly once with grad; its detached output reconstructs the input
        and the same graph gives the VJP (and accumulates the block's parameter grads).
        """
        gp = torch.zeros_like(p) if gp is None else gp
        gq = torch.zeros_like(q) if gq is None else gq
        if self.kind == "hamiltonian":                       # p' = p + F(q); q' = q + G(p')
            z = p.detach().requires_grad_()
            with torch.enable_grad():
                g = blk.mlp(z)
            q_in = q - g.detach()
            torch.autograd.backward(g, gq)
            gp = gp + z.grad                                  # total grad wrt p'
            w = q_in.detach().requires_grad_()
            with torch.enable_grad():
                f = blk.attn(w, cos, sin)
            p_in = p - f.detach()
            torch.autograd.backward(f, gp)
            return p_in, q_in, gp, gq + w.grad
        if self.kind == "midpoint":                          # (p_{j-1}, p_j) -> (p_j, p_{j+1})
            a, h = self.a, self.h
            x = p.detach().requires_grad_()
            with torch.enable_grad():
                y = blk.delta(x, cos, sin)
            p_prev = (q - (1 - a) * p - h * y.detach()) / a
            torch.autograd.backward(y, h * gq)
            return p_prev, p, a * gq, gp + (1 - a) * gq + x.grad
        h = self.h                                           # leapfrog: (x, v) -> (x + h v', v')
        gv = gq + h * gp                                      # total grad wrt v'
        x = (p - h * q).detach().requires_grad_()
        with torch.enable_grad():
            y = blk.delta(x, cos, sin)
        v_in = q - h * y.detach()
        torch.autograd.backward(y, h * gv)
        return x.detach(), v_in, gp + x.grad, gv

    def output(self, p, q):
        if self.kind == "leapfrog":
            return p
        if self.kind == "midpoint" and self.a < 1:
            return q                  # latest state: a < 1 already damps the parasitic mode (root -a)
        return 0.5 * (p + q)          # hamiltonian streams / plain midpoint: average


def _amp_state(device_type):
    # cache_enabled=False matters: autocast caches each weight's low-precision cast for the whole
    # autocast region. RevStackFn.forward runs with grad disabled, so those cached casts carry no graph.
    # If backward() is called inside the same autocast region, the re-evaluation would reuse them and
    # every block weight would silently receive no gradient. Disabling the cache here forces fresh casts.
    if device_type not in ("cuda", "cpu", "mps"):
        return dict(device_type=device_type, enabled=False, cache_enabled=False)
    return dict(device_type=device_type, enabled=torch.is_autocast_enabled(device_type),
                dtype=torch.get_autocast_dtype(device_type), cache_enabled=False)


class RevStackFn(torch.autograd.Function):
    """Runs all blocks without saving activations; backward rebuilds each block's input from its output."""

    @staticmethod
    def forward(ctx, p, q, model):
        ctx.model = model
        ctx.amp = _amp_state(p.device.type)
        for blk in model.blocks:
            p, q = model.stepper.step(blk, p, q, model.cos, model.sin)
        ctx.save_for_backward(p, q)
        return p, q

    @staticmethod
    def backward(ctx, gp, gq):
        model = ctx.model
        p, q = ctx.saved_tensors
        with torch.autocast(**ctx.amp), torch.no_grad():
            for blk in reversed(model.blocks):
                p, q, gp, gq = model.stepper.backward_step(blk, p, q, gp, gq, model.cos, model.sin)
        return gp, gq, None


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
class LLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.residual in ("euler",) + REVERSIBLE, cfg.residual
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight            # tied embeddings
        cos, sin = rope_cache(cfg.seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        if cfg.residual in REVERSIBLE:
            d = DEFAULTS.get(cfg.residual, {})
            self.stepper = Stepper(cfg.residual, cfg.h if cfg.h is not None else d.get("h"),
                                   cfg.a if cfg.a is not None else d.get("a"))
        self.apply(self._init)
        for n, p in self.named_parameters():                # GPT-2 style scaled init of output projections
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def hidden(self, idx):
        x = self.embed(idx)
        if self.cfg.residual == "euler":
            for blk in self.blocks:
                x = x + blk.delta(x, self.cos, self.sin)
            return self.norm(x)
        p, q = self.stepper.init(x)
        if self.cfg.rev_backprop and torch.is_grad_enabled():
            p, q = RevStackFn.apply(p, q, self)
        else:
            for blk in self.blocks:
                p, q = self.stepper.step(blk, p, q, self.cos, self.sin)
        return self.norm(self.stepper.output(p, q))

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:
            return self.lm_head(h)
        return self.loss(h, targets)

    def loss(self, h, targets):
        """Cross-entropy that never materialises the full (B*T, vocab) fp32 logits tensor."""
        h, targets = h.flatten(0, 1), targets.flatten()
        chunk = self.cfg.ce_chunk
        if not chunk:
            return F.cross_entropy(self.lm_head(h).float(), targets)
        total = h.new_zeros((), dtype=torch.float32)
        for hs, ts in zip(h.split(chunk), targets.split(chunk)):
            if torch.is_grad_enabled():
                total = total + checkpoint(_ce_sum, hs, ts, self.lm_head.weight, use_reentrant=False)
            else:                                   # evaluation: chunk too, no checkpoint needed
                total = total + _ce_sum(hs, ts, self.lm_head.weight)
        return total / targets.numel()


def _ce_sum(h, t, w):
    return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")
