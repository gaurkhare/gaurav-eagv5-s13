"""Training loop that records final loss, throughput (tokens/s) and peak memory for each run."""
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch

from .data import Batches
from .model import LLM, ModelConfig


@dataclass
class TrainConfig:
    name: str = "baseline"
    data_dir: str = "data"
    out_dir: str = "results"
    batch_size: int = 32
    total_tokens: int = 50_000_000
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 50          # x eval_bs sequences of val.bin
    eval_bs: int = 16
    log_every: int = 20
    seed: int = 1337
    model: ModelConfig = field(default_factory=ModelConfig)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return None


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30, float("nan")
    return float("nan"), float("nan")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_optimizer(model, cfg, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), fused=device.type == "cuda")


def train_step(model, opt, scaler, x, y, dtype, device, grad_clip):
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return loss, gnorm


@torch.no_grad()
def evaluate(model, val, cfg, dtype, device):
    model.eval()
    losses = []
    for i in range(cfg.eval_batches):
        x, y = val.get(range(i * cfg.eval_bs, (i + 1) * cfg.eval_bs), device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            losses.append(model(x, y).item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: TrainConfig):
    device = get_device()
    dtype = amp_dtype(device)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = LLM(cfg.model).to(device)
    opt = make_optimizer(model, cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    train_data = Batches(os.path.join(cfg.data_dir, "train.bin"), cfg.model.seq_len, cfg.total_tokens)
    val_data = Batches(os.path.join(cfg.data_dir, "val.bin"), cfg.model.seq_len, seed=None)
    val_data.order.sort()

    tok_per_step = cfg.batch_size * cfg.model.seq_len
    steps = len(train_data) // cfg.batch_size
    warmup = max(1, int(cfg.warmup_frac * steps))

    def lr_at(s):
        if s < warmup:
            return cfg.lr * (s + 1) / warmup
        t = (s - warmup) / max(1, steps - warmup)
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))

    print(f"[{cfg.name}] device={device} amp={dtype} params={model.num_params()/1e6:.2f}M "
          f"residual={cfg.model.residual}{' (reversible backprop)' if cfg.model.residual != 'euler' and cfg.model.rev_backprop else ''} "
          f"B={cfg.batch_size} T={cfg.model.seq_len} steps={steps} tokens={steps*tok_per_step/1e6:.1f}M lr={cfg.lr:g}")

    history, evals = [], []
    reset_peak(device)
    t_start = time.time()
    t_train = 0.0            # time spent in train steps only (evals excluded)
    timed_tokens = 0
    t_mark, tok_mark = time.time(), 0
    running = None
    for step, (x, y) in enumerate(train_data.epoch(cfg.batch_size, device)):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, gnorm = train_step(model, opt, scaler, x, y, dtype, device, cfg.grad_clip)
        tok_mark += tok_per_step

        if (step + 1) % cfg.log_every == 0 or step == steps - 1:
            loss_v = loss.item()                     # implicit sync
            sync(device)
            dt = time.time() - t_mark
            if step >= cfg.log_every:                # first window includes warm-up / allocator growth
                t_train += dt
                timed_tokens += tok_mark
            tps = tok_mark / dt
            running = loss_v if running is None else 0.9 * running + 0.1 * loss_v
            history.append(dict(step=step + 1, tokens=(step + 1) * tok_per_step, loss=loss_v,
                                lr=lr_at(step), grad_norm=float(gnorm), tok_per_s=tps))
            if (step + 1) % (cfg.log_every * 5) == 0 or step == steps - 1:
                print(f"step {step+1:5d}/{steps} loss {loss_v:.4f} (ema {running:.4f}) "
                      f"lr {lr_at(step):.2e} gnorm {float(gnorm):.2f} {tps/1e3:.1f}k tok/s "
                      f"peak {peak_mem_gb(device)[0]:.2f} GB")
            t_mark, tok_mark = time.time(), 0

        if (step + 1) % cfg.eval_every == 0 or step == steps - 1:
            sync(device)
            t0 = time.time()
            vl = evaluate(model, val_data, cfg, dtype, device)
            evals.append(dict(step=step + 1, tokens=(step + 1) * tok_per_step, val_loss=vl))
            print(f"  eval @ {step+1}: val_loss {vl:.4f}")
            t_mark += time.time() - t0              # do not charge eval time to throughput

    wall = time.time() - t_start
    alloc, reserved = peak_mem_gb(device)
    last = history[-max(1, len(history) // 20):]    # mean of the last 5% of logged train losses
    result = dict(
        name=cfg.name, device=str(device),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else str(device),
        amp_dtype=str(dtype), params=model.num_params(), steps=steps,
        tokens_trained=steps * tok_per_step,
        final_train_loss=sum(h["loss"] for h in last) / len(last),
        final_val_loss=evals[-1]["val_loss"],
        tokens_per_s=timed_tokens / t_train if t_train else float("nan"),
        peak_mem_allocated_gb=alloc, peak_mem_reserved_gb=reserved,
        wall_time_min=wall / 60, config=asdict(cfg), history=history, evals=evals,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"{cfg.name}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{cfg.name}] final train {result['final_train_loss']:.4f} | val {result['final_val_loss']:.4f} | "
          f"{result['tokens_per_s']/1e3:.1f}k tok/s | peak alloc {alloc:.2f} GB (reserved {reserved:.2f}) | "
          f"{result['wall_time_min']:.1f} min")
    return result, model


def fits(model_cfg: ModelConfig, batch_size, steps=2):
    """True if `steps` full train steps (fwd + bwd + AdamW) run at this batch size without OOM."""
    device = get_device()
    dtype = amp_dtype(device)
    model = opt = x = loss = None
    try:
        model = LLM(model_cfg).to(device)
        opt = make_optimizer(model, TrainConfig(), device)
        scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
        x = torch.randint(0, model_cfg.vocab_size, (batch_size, model_cfg.seq_len + 1), device=device)
        for _ in range(steps):
            loss, _ = train_step(model, opt, scaler, x[:, :-1], x[:, 1:], dtype, device, 1.0)
        sync(device)
        return True, peak_mem_gb(device)[0]
    except torch.OutOfMemoryError:
        return False, None
    finally:
        del model, opt, x, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        reset_peak(device)


def find_max_batch(model_cfg: ModelConfig, start=8, limit=4096, multiple=8):
    """Doubling then binary search for the largest batch (a multiple of `multiple`) that trains without OOM."""
    reset_peak(get_device())
    ok, peak = fits(model_cfg, start)
    if not ok:
        raise RuntimeError(f"batch {start} does not fit")
    lo, trials = start, {start: peak}
    hi = None
    while lo * 2 <= limit:
        ok, peak = fits(model_cfg, lo * 2)
        trials[lo * 2] = peak
        print(f"  B={lo*2:5d}: {'ok  peak %.2f GB' % peak if ok else 'OOM'}")
        if not ok:
            hi = lo * 2
            break
        lo *= 2
    if hi is None:
        return lo, trials
    while hi - lo > multiple:
        mid = (lo + hi) // 2 // multiple * multiple
        ok, peak = fits(model_cfg, mid)
        trials[mid] = peak
        print(f"  B={mid:5d}: {'ok  peak %.2f GB' % peak if ok else 'OOM'}")
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo, trials
