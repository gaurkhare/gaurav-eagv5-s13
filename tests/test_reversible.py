"""Reversible backprop must give the same gradients as ordinary autograd on the same architecture."""
import copy

import pytest
import torch

from revllm.model import LLM, ModelConfig


VARIANTS = {
    "hamiltonian": dict(residual="hamiltonian"),
    "midpoint_a0.5": dict(residual="midpoint", a=0.5, h=0.25),
    "midpoint_plain": dict(residual="midpoint", a=1.0, h=0.5),
    "leapfrog": dict(residual="leapfrog", h=1.0),
}


def small(variant, rev_backprop):
    kw = VARIANTS.get(variant, dict(residual=variant))
    return ModelConfig(vocab_size=97, seq_len=16, d_model=32, n_layers=4, n_heads=4, d_ff=64,
                       rev_backprop=rev_backprop, ce_chunk=20, **kw)


@pytest.mark.parametrize("residual", list(VARIANTS))
def test_grads_match_autograd(residual):
    torch.manual_seed(0)
    ref = LLM(small(residual, False)).double()
    rev = copy.deepcopy(ref)
    rev.cfg.rev_backprop = True
    x = torch.randint(0, 97, (3, 17))
    ref(x[:, :-1], x[:, 1:]).backward()
    loss = rev(x[:, :-1], x[:, 1:])
    loss.backward()
    for (n, a), b in zip(ref.named_parameters(), rev.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-10, rtol=1e-8), n


@pytest.mark.parametrize("residual", list(VARIANTS))
def test_inverse_reconstructs_input(residual):
    torch.manual_seed(0)
    m = LLM(small(residual, True))
    x = m.embed(torch.randint(0, 97, (2, 16)))
    with torch.no_grad():
        p, q = m.stepper.init(x)
        p0, q0 = p.clone(), q.clone()
        for blk in m.blocks:
            p, q = m.stepper.step(blk, p, q, m.cos, m.sin)
        for blk in reversed(m.blocks):
            p, q = m.stepper.inverse(blk, p, q, m.cos, m.sin)
    assert (p - p0).abs().max() < 1e-5 and (q - q0).abs().max() < 1e-5


def test_chunked_ce_matches_full():
    torch.manual_seed(0)
    m = LLM(small("euler", False)).double()
    x = torch.randint(0, 97, (3, 17))
    a = m(x[:, :-1], x[:, 1:])
    m.cfg.ce_chunk = 0
    b = m(x[:, :-1], x[:, 1:])
    assert torch.allclose(a, b)


@pytest.mark.parametrize("backward_inside_autocast", [False, True])
@pytest.mark.parametrize("residual", list(VARIANTS))
def test_autocast_grads_present_and_close(residual, backward_inside_autocast):
    """Under bf16 autocast every parameter must get a gradient, whether backward() runs inside or
    outside the autocast region. Regression test: autocast's weight-cast cache used to leak the
    no-grad casts from RevStackFn.forward into the backward re-evaluation, dropping all block grads."""
    torch.manual_seed(0)
    ref = LLM(small(residual, False))
    rev = copy.deepcopy(ref)
    rev.cfg.rev_backprop = True
    x = torch.randint(0, 97, (3, 17))
    for m in (ref, rev):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = m(x[:, :-1], x[:, 1:])
            if backward_inside_autocast:
                loss.backward()
        if not backward_inside_autocast:
            loss.backward()
    missing = [n for n, p in rev.named_parameters() if p.grad is None]
    assert not missing, missing
    g1 = torch.cat([p.grad.flatten() for p in ref.parameters()])
    g2 = torch.cat([p.grad.flatten() for p in rev.parameters()])
    assert ((g1 - g2).norm() / g1.norm()).item() < 0.1
