# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: incorrectly_encoded_metadata,ablation,title,x,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python (ssm-quantization)
#     language: python
#     name: ssm-quantization
# ---

# %% [markdown]
# # SSM Quantization Stability (Deep Learning Course Project, Spring2026)
#
# ## Code
#
# This notebook contains the code to run experiments for our project.
#
# ## Attribution
# `S4Model` is taken from the S4 repository
# (HazyResearch/state-spaces, Apache-2.0; `s4/example.py`); `discretize`, the conv
# FFT, and the `SSMLayer` recurrence are adapted from `s4/models/s4/s4d.py` and
# `s4/models/s4/s4.py`.
#

# %% Setup
import copy
import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib
if "ipykernel" not in sys.modules:         
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # MPS excluded (complex)
print("device:", device)


# %% [markdown]
# ## Quantizers
#
# Zhao et al. implement fake-quantization described in Jacob et al. (2018). You can read algorithms of different quantizers, but most important are:
# - `fake_quant_complex_asym` aka `cartesian-asym (the scheme Zhao et al. use)
# - `fake_quant_logpolar`/`fake_quant_logpolar_r1` aka `logpolar`/`logpolar-r1` (the scheme we propose)

# %% Quantizers
def fake_quant(x, bits, per_head=True):
    """Uniform symmetric fake-quantization of a REAL tensor, optionally per S4D head. 
        Maps all values into grid [-qmax-1, qmax] with cells `scale`.
        Notice that this scheme always has a zero point 
        (torch.round(x/scale) = 0 for abs(x) < scale)"""
    maxval = x.abs().amax(dim=-1, keepdim=True) if per_head else x.abs().amax()
    qmax = 2 ** (bits - 1) - 1
    scale = (maxval / qmax).clamp(min=1e-12)
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def fake_quant_complex(z, bits):
    return torch.complex(fake_quant(z.real, bits), fake_quant(z.imag, bits))


def fake_quant_asym(x, bits, per_head=True):
    """Asymmetric fake-quantization of a REAL tensor, optionally per S4D head.
        Fits the grid to the OBSERVED range [lo, hi] = [min, max] instead of a symmetric
        [-max, max], so it wastes no cells when the values are one-sided -- e.g. Re(A)
        clusters near +1, where a symmetric grid would spend half its levels on the empty
        negative side. .
        The integer zero-point zp = round(-lo/scale) shifts the grid so that x = 0 lands
        a grid point (q = zp)"""
    if per_head:
        lo, hi = x.amin(dim=-1, keepdim=True), x.amax(dim=-1, keepdim=True)
    else:
        lo, hi = x.amin(), x.amax()
    levels = 2 ** bits - 1
    scale = ((hi - lo) / levels).clamp(min=1e-12)
    zp = torch.round(-lo / scale)
    q = torch.clamp(torch.round(x / scale) + zp, 0, levels)
    return (q - zp) * scale


def fake_quant_complex_asym(z, bits):
    return torch.complex(fake_quant_asym(z.real, bits), fake_quant_asym(z.imag, bits))


def quant_uniform(x, lo, hi, bits):
    """Uniform grid on [lo, hi] (scalars or broadcastable per-head tensors).
        Has zero point (when abs(x -lo) < scale )"""
    levels = 2 ** bits - 1
    scale = torch.as_tensor((hi - lo) / levels, dtype=x.dtype, device=x.device).clamp(min=1e-12)
    q = torch.clamp(torch.round((x - lo) / scale), 0, levels)
    return lo + q * scale


def quant_phase(theta, bits, per_head=True):
    """Phase quantized over its OBSERVED per-head range (puts theta=0 on the grid and 
    doesnt waste bits for values that are never realized by theta"""
    if per_head:
        t_lo, t_hi = theta.amin(dim=-1, keepdim=True), theta.amax(dim=-1, keepdim=True)
    else:
        t_lo, t_hi = theta.amin(), theta.amax()
    return quant_uniform(theta, t_lo, t_hi, bits)


def fake_quant_polar(z, bits=None, eps=1e-6, per_head=True, calibrate_phase=True,
                     bits_r=None, bits_theta=None):
    """Polar: quantize magnitude and
    phase separately. bits_r/bits_theta override `bits` per coordinate.
    `bits_r == bits_theta == None`: each of the parameters quantized to `bits`"""
    if bits_r is None and bits_theta is None:
        bits_r = bits_theta = bits
    r, theta = z.abs(), torch.angle(z)
    if bits_r is not None:
        r_hi = r.amax(dim=-1, keepdim=True) if per_head else r.amax().clamp(max=1 - eps)
        r = quant_uniform(r, 0.0, r_hi, bits_r)
    if bits_theta is not None:
        theta = quant_phase(theta, bits_theta, per_head) if calibrate_phase \
            else quant_uniform(theta, -math.pi, math.pi, bits_theta)
    return r * torch.exp(1j * theta)


def fake_quant_logpolar(z, bits=None, eps=1e-6, per_head=True, calibrate_phase=True,
                        bits_r=None, bits_theta=None):
    """Log-polar: quantize s = log(1-r) instead of r. You can read in our report why this works so
    well for A"""
    if bits_r is None and bits_theta is None:
        bits_r = bits_theta = bits
    r, theta = z.abs(), torch.angle(z)
    if bits_r is not None:
        s = torch.log((1 - r).clamp(min=eps))
        if per_head:
            s_lo, s_hi = s.amin(dim=-1, keepdim=True), s.amax(dim=-1, keepdim=True)
        else:
            s_lo, s_hi = s.amin(), s.amax()
        r = 1 - torch.exp(quant_uniform(s, s_lo, s_hi, bits_r))
    if bits_theta is not None:
        theta = quant_phase(theta, bits_theta, per_head) if calibrate_phase \
            else quant_uniform(theta, -math.pi, math.pi, bits_theta)
    return r * torch.exp(1j * theta)


def fake_quant_polar_bad(z, bits):       
    """Uncalibrated [-pi,pi] phase grid"""
    return fake_quant_polar(z, bits, calibrate_phase=False)


def fake_quant_logpolar_bad(z, bits):
    """Uncalibrated [-pi,pi] phase grid"""
    return fake_quant_logpolar(z, bits, calibrate_phase=False)


def fake_quant_complex_proj(z, bits, eps=1e-4):
    """Cartesian quantization, bring |z_q|>=1 back onto the disk.
    Fixes instability but doesn't fix resolution mismatch (most of the bits go to space that isn't
    realized)"""
    zq = fake_quant_complex(z, bits)
    r = zq.abs()
    mask = r >= 1.0
    if mask.any():
        zq = torch.where(mask, zq / r.clamp(min=1e-12) * (1 - eps), zq)
    return zq


def fake_quant_magonly(z, bits):    return fake_quant_polar(z, bits_r=bits)     # phase exact
def fake_quant_logmagonly(z, bits): return fake_quant_logpolar(z, bits_r=bits)
def fake_quant_phaseonly(z, bits):  return fake_quant_polar(z, bits_theta=bits)  # magnitude exact
def fake_quant_logpolar_r1(z, bits): return fake_quant_logpolar(z, bits_r=1, bits_theta=2 * bits - 1)


def fake_quant_logpolar_r0(z, bits, eps=1e-6):
    """Zero radius bits, it's a constant that equals to arithmetic average of min and max. The whole budget goes to phase."""
    r, theta = z.abs(), torch.angle(z)
    s = torch.log((1 - r).clamp(min=eps))
    s_mid = (s.amin(dim=-1, keepdim=True) + s.amax(dim=-1, keepdim=True)) / 2
    return (1 - torch.exp(s_mid)) * torch.exp(1j * quant_phase(theta, 2 * bits, per_head=True))


def fake_quant_unitary(z, bits):
    """r = 1 for every mode (orthogonal-RNN limit): pure rotation, whole budget to phase."""
    return torch.polar(torch.ones_like(z.abs()), quant_phase(torch.angle(z), 2 * bits, per_head=True))


QUANTIZERS = {
    "cartesian": fake_quant_complex, "cartesian-asym": fake_quant_complex_asym,
    "cartesian-proj": fake_quant_complex_proj, "polar": fake_quant_polar,
    "logpolar": fake_quant_logpolar, "polar-bad": fake_quant_polar_bad,
    "logpolar-bad": fake_quant_logpolar_bad, "mag-only": fake_quant_magonly,
    "logmag-only": fake_quant_logmagonly, "phase-only": fake_quant_phaseonly,
    "logpolar-r1": fake_quant_logpolar_r1, "logpolar-r0": fake_quant_logpolar_r0,
    "unitary": fake_quant_unitary,
}
print("schemes:", list(QUANTIZERS))


# %% [markdown]
# ## State quantization (dynamic per-head asymmetric)
#
# The state is an activation: quantize it per forward with fake_quant_complex_asym, whose
# per-head range is read live from the current state (dynamic), so nothing is clipped away.

# %%
def quant_state(x, bits):
    return fake_quant_complex_asym(x, bits)

def discretize(kernel):
    """ZOH discretization of the S4D kernel -> (dA, dB, dC).
    dA = exp(A*dt); dB = (dA-1)/A (folds B=1); dC from the complex C parameter.
    """
    dt = torch.exp(kernel.log_dt)
    A = -torch.exp(kernel.log_A_real) + 1j * kernel.A_imag
    dA = torch.exp(A * dt.unsqueeze(-1))
    dB = (dA - 1.) / A
    dC = torch.view_as_complex(kernel.C)
    return dA, dB, dC




# %% [markdown]
# ## QuantConfig
#
# This config contains all quantization parameters we use 

# %%
@dataclass
class QuantConfig:
    """All quantization knobs for one forward pass (fp when every field is left default)."""
    scheme: object = None        # Ā grid: a name in QUANTIZERS, a callable, or None (fp)
    a_bits: int = None           # bits for Ā (per coordinate); None = fp
    bcd_bits: int = None         # bits for B̄/C̄/D (asymmetric); None = fp
    x_bits: int = None           # bits for the recurrent state; None = fp



# %% [markdown]
# ## Model (S4Model backbone; the S4D layer is the vendored library)

# %% S4Model
sys.path.insert(0, os.path.join(os.getcwd(), "s4"))
from models.s4.s4d import S4D

class S4Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=16, n_layers=4, dropout=0.1, prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers, self.norms, self.dropouts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=0.001))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):                    # the vendored conv-mode forward (for loading only)
        x = self.encoder(x).transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            x = dropout(z) + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        return self.decoder(x.transpose(-1, -2).mean(dim=1))


# %%
from torchinfo import summary
m = S4Model(1, 10, 16, 4)                 # d_input=1, d_output=10, d_model=16, n_layers=4
summary(m, input_size=(1, 784, 1),        # (batch, seq_len, d_input)
        col_names=("input_size", "output_size", "num_params"),
        depth=4, device="cpu")

# %% [markdown]
# ## PTQ+QAT machinery

# %% QAT machinery
CAPPED_SCHEMES = ("polar", "logpolar", "polar-bad", "logpolar-bad")

def apply_q(fn, x, *args):
    """Quantize x with fn but pass gradients straight through (STE): forward = fn(x),
    backward = identity. """
    return x + (fn(x.detach(), *args) - x).detach()

def cap_disk(z, eps=1e-4):
    r = z.abs()
    return torch.where(r >= 1 - eps, z / r.clamp(min=1e-12) * (1 - eps), z)


def _ssm_scan(dA, dB, dC, u, x_bits=None, diag=None, trace=None, observe=None):
    """The _ssm_scan function adapted from S4 repo"""
    B, H, L = u.shape
    x = torch.zeros(B, H, dA.shape[-1], dtype=torch.cfloat, device=u.device)
    ys = []
    for t in range(L):
        x = dA * x + dB * u[..., t].unsqueeze(-1) # can do dA * x because dA is diagonal
        if x_bits is not None:
            x = apply_q(quant_state, x, x_bits)
        if observe is not None:
            observe(x)
        if diag is not None:
            m = x.abs().amax().item()
            diag["maxX"] = max(diag.get("maxX", 0.0), m if math.isfinite(m) else 1e30)
        if trace is not None:
            trace.append(min(x.abs().amax().item(), 1e30))
        ys.append(2 * torch.einsum('hn,bhn->bh', dC, x).real)
    return torch.stack(ys, -1)


class SSMLayer(nn.Module):
    """One diagonal S4D block as explicit discrete matrices (dA, dB, dC, D) + GLU tail.

    Single implementation for PTQ (eval) and QAT (training).
    The matrices are stored fp and quantized per-forward from a QuantConfig, so one model
    sweeps every scheme. `train_a` makes Ā trainable (with quantization gradient computed as an identity function, method
    known as STE - straight-throught gradient estimator)"""

    def __init__(self, layer, train_a=False):
        super().__init__()
        self.train_a = train_a
        with torch.no_grad():
            dA, dB, dC = discretize(layer.kernel)
        self.dA = nn.Parameter(torch.view_as_real(dA.clone()), requires_grad=train_a)
        self.dB = nn.Parameter(torch.view_as_real(dB.clone()))
        self.dC = nn.Parameter(torch.view_as_real(dC.clone()))
        self.D = nn.Parameter(layer.D.detach().clone())
        self.activation, self.dropout, self.output_linear = \
            layer.activation, layer.dropout, layer.output_linear

    def matrices(self, cfg):
        """Quantized (dA, dB, dC, D) for one forward. """
        dA = torch.view_as_complex(self.dA)
        if self.train_a and cfg.scheme in CAPPED_SCHEMES:
            dA = cap_disk(dA)
        if cfg.scheme is not None and cfg.a_bits is not None:
            q = QUANTIZERS[cfg.scheme] if isinstance(cfg.scheme, str) else cfg.scheme
            dA = apply_q(q, dA, cfg.a_bits)
        dB, dC, D = torch.view_as_complex(self.dB), torch.view_as_complex(self.dC), self.D
        if cfg.bcd_bits is not None:
            dB = apply_q(fake_quant_complex_asym, dB, cfg.bcd_bits)
            dC = apply_q(fake_quant_complex_asym, dC, cfg.bcd_bits)
            D = apply_q(fake_quant_asym, D, cfg.bcd_bits, False)   # per_head=False
        return dA, dB, dC, D

    def forward(self, u, cfg, mode="rec", diag=None, trace=None, observe=None):
        """The forward function adapted from S4 repo"""

        dA, dB, dC, D = self.matrices(cfg)
        if diag is not None:
            diag["maxA"] = max(diag.get("maxA", 0.0), dA.abs().amax().item())
        if mode == "conv":
            L = u.size(-1)
            lpow = dA.unsqueeze(-1) ** torch.arange(L, device=u.device)
            K = 2 * torch.einsum('hn,hnl->hl', dC * dB, lpow).real
            y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(K, n=2 * L), n=2 * L)[..., :L]
        else:
            y = _ssm_scan(dA, dB, dC, u, x_bits=cfg.x_bits, diag=diag, trace=trace, observe=observe)
        return self.output_linear(self.dropout(self.activation(y + u * D.unsqueeze(-1))))


class SSMModel(nn.Module):
    """The whole S4D classifier as SSMLayers. One forward serves both eval and training,
    conv or recurrent. Built from a loaded S4Model (which holds the trained weights)."""

    def __init__(self, base, train_a=False):
        super().__init__()
        b = copy.deepcopy(base)
        self.encoder, self.decoder = b.encoder, b.decoder
        self.norms, self.dropouts = b.norms, b.dropouts
        self.layers = nn.ModuleList(SSMLayer(l, train_a) for l in b.s4_layers)

    def forward(self, u, cfg=None, mode="rec", diag=None, trace=None, observe=None):
        cfg = cfg if cfg is not None else QuantConfig()
        x = self.encoder(u).transpose(-1, -2)
        for i, (layer, norm, drop) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            obs = observe[i] if isinstance(observe, (list, tuple)) else observe
            z = layer(x, cfg, mode=mode, diag=diag, trace=trace, observe=obs)
            x = norm((drop(z) + x).transpose(-1, -2)).transpose(-1, -2)
        return self.decoder(x.transpose(-1, -2).mean(dim=1))

    @torch.no_grad()
    def max_absA(self, cfg=None):
        cfg = cfg if cfg is not None else QuantConfig()
        return max(l.matrices(cfg)[0].abs().amax().item() for l in self.layers)


RESULTS_CSV = "results/experiments.csv"
_CSV_FIELDS = ["name", "phase", "scheme", "a_bits", "bcd_bits", "x_bits",
               "lr", "epochs", "acc", "peak_absA", "maxA", "maxX"]

def log_result(name, phase, cfg, **metrics):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    scheme = cfg.scheme if isinstance(cfg.scheme, str) else getattr(cfg.scheme, "__name__", repr(cfg.scheme))
    row = dict.fromkeys(_CSV_FIELDS, "")
    row.update(name=name, phase=phase,
               scheme=scheme, a_bits=cfg.a_bits, bcd_bits=cfg.bcd_bits, x_bits=cfg.x_bits, **metrics)
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return row


@torch.no_grad()
def accuracy(name, model, loader, limit=1000, cfg=None, mode="rec"):
    """Test accuracy (%) + the diag dict (maxA / maxX); logs a PTQ row named `name`."""
    model.eval()
    correct = total = 0
    diag = {}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb, cfg=cfg, mode=mode, diag=diag)
        correct += (out.argmax(1) == yb).sum().item()
        total += yb.size(0)
        if total >= limit:
            break
    acc = 100.0 * correct / total
    cfg = cfg if cfg is not None else QuantConfig()
    log_result(name, "ptq", cfg, acc=round(acc, 3),
               maxA=round(diag.get("maxA", float("nan")), 6),   # maxX absent in conv mode: expected
               maxX=round(diag.get("maxX", float("nan")), 4))
    return acc, diag


def train_qat(name, model, loader, cfg, epochs, lr, train_limit):
    """Fine-tune (no gradient clipping). Returns peak max|A_bar_q| and logs a QAT-train row.
    Only requires_grad leaves are optimized, so frozen-Ā vs trainable-Ā is one flag
    on the model (SSMLayer.train_a)."""
    model.train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    crit = nn.CrossEntropyLoss()
    maxA_peak = model.max_absA(cfg)
    for ep in range(epochs):
        seen = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            crit(model(inputs, cfg=cfg), targets).backward()
            opt.step()
            maxA_peak = max(maxA_peak, model.max_absA(cfg))
            seen += targets.size(0)
            if train_limit is not None and seen >= train_limit:
                break
    log_result(name, "qat_train", cfg, lr=lr, epochs=epochs, peak_absA=round(maxA_peak, 6))
    return maxA_peak


SCHEMES_MAIN = ["cartesian-asym", "cartesian-proj", "polar", "logpolar", "logpolar-r1", "unitary"]


# %% [markdown]
# B. Load data + model

# %% Data + model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(1, 784).t()),
])
testset = torchvision.datasets.MNIST("s4/data", train=False, download=True, transform=transform)
trainset = torchvision.datasets.MNIST("s4/data", train=True, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=250, num_workers=0)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

base = S4Model(1, 10, 16, 4, 0.1, False).to(device)      # vendored backbone, holds the trained weights
ckpt = torch.load("s4/checkpoint/ckpt.pth", map_location="cpu")
base.load_state_dict(ckpt["model"])
base.eval()
model = SSMModel(base).to(device)                        # unified PTQ/QAT model over discrete matrices
model.eval()
print(f"loaded baseline: val acc {ckpt['acc']:.2f}% @ epoch {ckpt['epoch']}")





# %% [markdown]
# ## Eigenvalue distribution + stability table
#
#

# %% Eigenvalue distribution
print(f"let N=1/(1-r) (memory time constant)")
for i, layer in enumerate(model.layers):
    r = layer.matrices(QuantConfig())[0].abs().flatten()
    
    N = 1 / (1 - r).clamp(min=1e-9)
    
    b = [(N < 10).float().mean(), ((N >= 10) & (N < 100)).float().mean(),
         ((N >= 100) & (N < 1000)).float().mean(), (N >= 1000).float().mean()]
    print(f"layer {i}: fast {b[0]:.0%} (N<10) med {b[1]:.0%} (100>N>=10) slow {b[2]:.0%} (1000>N>=100) v.slow {b[3]:.0%} (N>=1000) "
          f"| |A| median {r.median():.4f} max {r.max():.5f}")

# %% Eigenvalue distribution
fig, ax = plt.subplots(figsize=(6, 6))

# unit circle + axes (the stability boundary)
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), 'k-', lw=1)
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)

colors = plt.cm.viridis(np.linspace(0, 1, len(model.layers)))
for i, layer in enumerate(model.layers):
    dA = layer.matrices(QuantConfig())[0].detach().cpu().numpy()   # (H, N) complex
    ax.scatter(dA.real.ravel(), dA.imag.ravel(),                   # all H*N eigenvalues, one color
               s=12, color=colors[i], alpha=0.7, label=f"layer {i}")
    # optional: the conjugate half, same color

ax.set_aspect('equal')
ax.set_xlabel("Re(Ā)"); ax.set_ylabel("Im(Ā)")
ax.set_title("Ā eigenvalues by layer (conjugates not displayed)")
ax.legend()
plt.tight_layout(); plt.show()


# %% Eigenvalue distribution
n = len(model.layers)
fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharex=True, sharey=True)
th = np.linspace(0, 2 * np.pi, 400)
head_colors = plt.cm.tab20(np.linspace(0, 1, 16))          # 16 head colors

for i, (ax, layer) in enumerate(zip(axes, model.layers)):
    dA = layer.matrices(QuantConfig())[0].detach().numpy()   # (H, N) complex
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1)
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    for h in range(dA.shape[0]):
        ax.scatter(dA[h].real, dA[h].imag, s=14, color=head_colors[h])
    ax.set_aspect('equal'); ax.set_title(f"layer {i}"); ax.set_xlabel("Re(Ā)")

axes[0].set_ylabel("Im(Ā)")
fig.suptitle("Ā eigenvalues — facet by layer, color by head (conjugates not displayed)")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Maximum eigenvalue in different quantization schemes

# %% Stability table
print(f"{'scheme':>16} | " + " | ".join(f"{bb}b" for bb in [8, 6, 4]))
for s in SCHEMES_MAIN:
    print(f"{s:>16} | " + " | ".join(
        f"{model.max_absA(QuantConfig(scheme=s, a_bits=bb)):.4f}" for bb in [8, 6, 4]))


# %% [markdown]
# ## PTQ scheme comparison

# %% Scheme comparison
probe = next(iter(test_loader))[0][:32].to(device)
print(f"{'scheme':>16} | {'bits':>4} | {'acc%':>6} | {'max|A|':>7} | stable")
for s in ["(fp)"] + SCHEMES_MAIN:
    for bb in ([None] if s == "(fp)" else [8, 6, 4]):
        cfg = QuantConfig(scheme=(None if s == "(fp)" else s), a_bits=bb)
        acc, diag = accuracy(f"ptq_{s}_{bb}b", model, test_loader, cfg=cfg)
        mA = diag.get("maxA", model.max_absA(cfg))
        print(f"{s:>16} | {str(bb):>4} | {acc:>6.2f} | {mA:>7.4f} | {'yes' if mA <= 1 else 'NO'}")
cfg = QuantConfig(scheme="logpolar", a_bits=8)
with torch.no_grad():
    yc = model(probe, cfg=cfg, mode="conv")
    yr = model(probe, cfg=cfg, mode="rec")
print(f"conv vs rec (logpolar@8): logit diff {(yc - yr).abs().max():.2e}, "
      f"agreement {(yc.argmax(1) == yr.argmax(1)).float().mean():.0%}")


# %% [markdown]
# ## Only cartesian-asym vs logpolar-1

# %% Scheme comparison
probe = next(iter(test_loader))[0][:32].to(device)
print(f"{'scheme':>16} | {'bits':>4} | {'acc%':>6} | {'max|A|':>7} | stable")
for s in ["(fp)"] + ["cartesian-asym", "logpolar-r1"]:
    for bb in ([None] if s == "(fp)" else [8, 7, 6,5, 4]):
        cfg = QuantConfig(scheme=(None if s == "(fp)" else s), a_bits=bb)
        acc, diag = accuracy(f"ptq_{s}_{bb}b", model, test_loader, cfg=cfg)
        mA = diag.get("maxA", model.max_absA(cfg))
        print(f"{s:>16} | {str(bb):>4} | {acc:>6.2f} | {mA:>7.4f} | {'yes' if mA <= 1 else 'NO'}")
cfg = QuantConfig(scheme="logpolar", a_bits=8)
with torch.no_grad():
    yc = model(probe, cfg=cfg, mode="conv")
    yr = model(probe, cfg=cfg, mode="rec")


# %% [markdown]
# ## Streaming divergence (max|state| over 4x sequence length)

# %% Divergence
xin = model.encoder(probe[:16].repeat(1, 4, 1)).transpose(-1, -2)
plt.figure(figsize=(8, 5))
for label, s, b in [("fp", None, None), ("cartesian 8b", "cartesian", 8),
                    ("cartesian 6b", "cartesian", 6), ("polar 6b", "polar", 6),
                    ("logpolar 6b", "logpolar", 6)]:
    tr = []
    with torch.no_grad():
        model.layers[0](xin, QuantConfig(scheme=s, a_bits=b), mode="rec", trace=tr)
    plt.semilogy(tr, label=label)
plt.axvline(784, color="gray", ls="--", label="train horizon")
plt.xlabel("timestep"); plt.ylabel("max |state|"); plt.legend(); plt.title("streaming divergence")
plt.tight_layout(); plt.show()


# %% Divergence
import copy, torch
import torch.nn as nn

xb, yb = next(iter(train_loader)); xb, yb = xb.to(device), yb.to(device)
crit = nn.CrossEntropyLoss()

def probe_layer_no_norm(target_max, k=3, L=784):
    layer = copy.deepcopy(model.layers[k]).to(device)
    layer.requires_grad_(True)
    with torch.no_grad():                                        # destabilize this layer's Ā
        d = torch.view_as_complex(layer.dA); d.mul_(target_max / d.abs().amax())
        layer.dA.copy_(torch.view_as_real(d))
    u = model.encoder(xb[:, :L, :]).transpose(-1, -2).detach()   # layer input (B,H,L)
    y = layer(u, QuantConfig())                                  # SSMLayer.forward -> (B,H,L), NO norm
    logits = model.decoder(y.transpose(-1, -2).mean(dim=1))      # (B,H,L)->(B,L,H)->mean->(B,H)->(B,10)
    loss = crit(logits, yb)                                      # cross-entropy
    layer.zero_grad(set_to_none=True)
    loss.backward()
    return layer.dA.grad.norm().item()

def probe_layer_norm(target_max, k=3, L=784):
    m = copy.deepcopy(model)                 # fresh copy; train_a stays False -> no cap_disk
    m.requires_grad_(True)
    with torch.no_grad():                    # rescale layer-0 eigenvalues to max|A| = target_max
        d = torch.view_as_complex(m.layers[k].dA)
        d.mul_(target_max / d.abs().amax())
        m.layers[0].dA.copy_(torch.view_as_real(d))
    diag = {}
    m.zero_grad(set_to_none=True)
    xin = xb[:, :L, :]
    logits = m(xin, cfg=QuantConfig(), diag=diag)
    loss = crit(logits, yb)
    loss.backward()
    return diag.get("maxX"), m.layers[0].dA.grad.norm().item()

print("--- sweep max|A|---")
for s in [0.99, 0.999, 1.0, 1.003, 1.01, 1.05,1.06]:
    k=3
    maxX, g = probe_layer_norm(s, k=k)
    print(f"max|A|={s:.3f}  maxX={maxX:.2e}  ||dA.grad||(with norm+res+drop)={g:.3e} "
          f"||dA.grad||(raw)={probe_layer_no_norm(s, k=k):.3e}")


# %% [markdown]
# ## Coordinate ablations (projection / attribution / bit-split) for $x$ in full precision

# %% Projection + attribution
print("projection:")
for s in ["cartesian", "cartesian-proj", "polar", "logpolar"]:
    print(f"  {s:>15}  8/6/5/4b: " +
          "/".join(f"{accuracy(f'ptq_{s}_{b}b', model, test_loader, cfg=QuantConfig(scheme=s, a_bits=b))[0]:.1f}"
                   for b in [8, 6, 5, 4]))
print("attribution:")
for s in ["mag-only", "logmag-only", "phase-only"]:
    print(f"  {s:>15}  8/6/4/3b: " +
          "/".join(f"{accuracy(f'ptq_{s}_{b}b', model, test_loader, cfg=QuantConfig(scheme=s, a_bits=b))[0]:.1f}"
                   for b in [8, 6, 4, 3]))

# %% [markdown]
# ## `cartesian-asym` vs `logpolar` for quantized $x$

# %%
print("PTQ: A=16/8/7/6 x=16/8/7/6 ")
for s in ["cartesian-asym", "logpolar"]:
    for a in [16,8,7,6]:
        for x in [16,8,7,6]:
            res = accuracy(f"ptq_{s}_A{a}x{x}", model, test_loader, cfg=QuantConfig(scheme=s, a_bits=a, x_bits=x))[0]
            print(f"{s} A{a}x{x}: {res:.3f}%")


# %% [markdown]
# ## logpolar-r1 (one magnitude bit, all the other go to phase)

# %%
print("PTQ logpolar: A=6/5/4 x=16 ")
for s in ["logpolar"]:
    for a in [6,5,4]:
        for x in [16]:
            res = accuracy(f"ptq_{s}_A{a}x{x}", model, test_loader, cfg=QuantConfig(scheme=s, a_bits=a, x_bits=x))[0]
            print(f"{s} A{a}x{x}: {res:.3f}%")

print("PTQ logpolar-r1: A=6/5/4/3/2 x=16 ")
for s in ["logpolar-r1"]:
    for a in [6,5,4,3,2]:
        for x in [16]:
            res = accuracy(f"ptq_{s}_A{a}x{x}", model, test_loader, cfg=QuantConfig(scheme=s, a_bits=a, x_bits=x))[0]
            print(f"{s} A{a}x{x}: {res:.3f}%")

# %% Bit-split sweep (fixed total budget)
for total in [16, 12, 10, 8]:
    results = []
    for rb in range(2, total - 1, 2):            # magnitude bits: 2, 4, ... ; phase gets the rest
        tb = total - rb
        scheme = lambda z, _bits, rb=rb, tb=tb: fake_quant_logpolar(z, bits_r=rb, bits_theta=tb)
        cfg = QuantConfig(scheme=scheme, a_bits=1)   # a_bits unused; just triggers the quant branch
        acc = accuracy(f"split_lp_r{rb}t{tb}", model, test_loader, cfg=cfg)[0]
        results.append((rb, tb, acc))

    best_rb, best_tb, best_acc = max(results, key=lambda r: r[2])
    print(f"logpolar total={total:>2}b: best r{best_rb}/t{best_tb} -> {best_acc:.1f}%")



# %% [markdown]
# ## QAT — frozen and trainable-A arms

# %% QAT runs
def run_qat(name, scheme, a_bits, x_bits, train_a=False, lr=1e-3, epochs=2, train_limit=6400):
    m = SSMModel(base, train_a=train_a).to(device)
    cfg = QuantConfig(scheme=scheme, a_bits=a_bits, bcd_bits=4, x_bits=x_bits)
    peak = train_qat(name, m, train_loader, cfg, epochs, lr, train_limit)
    acc = accuracy(f"{name}_eval", m, test_loader, 1000, cfg)[0]
    return acc, peak



# %%
print("Frozen A, cartesian-asym, x=16/8/7/6  (QAT acc):")
print(f"{'A/x':>7} | {'acc':>6}")
for a_bits in [16, 8, 7, 6]:
    for x_bits in [16, 8, 7, 6]:
        acc, _ = run_qat(f"qat_frozen_cartesian-asym_A{a_bits}x{x_bits}", "cartesian-asym",
                         a_bits, x_bits)
        print(f"A{a_bits}/x{x_bits} | {acc:>6.1f}")

print("Frozen A, logpolar, x=16/8/7/6  (QAT acc):")
print(f"{'A/x':>7} | {'acc':>6}")
for a_bits in [16, 8, 7, 6]:
    for x_bits in [16, 8, 7, 6]:
        acc, _ = run_qat(f"qat_frozen_logpolar_A{a_bits}x{x_bits}", "logpolar", a_bits, x_bits)
        print(f"A{a_bits}/x{x_bits} | {acc:>6.1f}")

print("Frozen A, logpolar-r1, x=16/8/7/6  (QAT acc):")
print(f"{'A/x':>7} | {'acc':>6}")
for a_bits in [16, 8, 7, 6]:
    for x_bits in [16, 8, 7, 6]:
        acc, _ = run_qat(f"qat_frozen_logpolar_A{a_bits}x{x_bits}", "logpolar", a_bits, x_bits)
        print(f"A{a_bits}/x{x_bits} | {acc:>6.1f}")


# %%
print("Trainable A, logpolar/logpolar-r1, x=7/6:")
for scheme in ["logpolar", "logpolar-r1"]:
    print(scheme)
    print(f"{'X bits':>6} | {'acc':>6} | {'peak|A|':>8}")
    for a_bits in [16, 8, 7, 6]:
        for x_bits in [16, 8, 7, 6]:
            acc, peak = run_qat(f"qat_trainA_{scheme}_A7x{x_bits}", scheme, a_bits=7,
                                x_bits=x_bits, train_a=True)
            print(f"X{x_bits:>5} | {acc:>6.1f} | {peak:>8.4f}")


# %%
def probe_peakA(scheme, a_bits, x_bits=6, train_a=True,
                lr=1e-3, epochs=2, train_limit=6400):
    m = SSMModel(base, train_a=train_a).to(device)
    cfg = QuantConfig(scheme=scheme, a_bits=a_bits, bcd_bits=4, x_bits=x_bits)
    opt = torch.optim.Adam((p for p in m.parameters() if p.requires_grad), lr=lr)
    crit = nn.CrossEntropyLoss()
    m.train()
    traj = [m.max_absA(cfg)]
    for ep in range(epochs):
        seen = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            crit(m(inputs, cfg=cfg), targets).backward()
            opt.step()
            traj.append(m.max_absA(cfg))
            seen += targets.size(0)
            if seen >= train_limit:
                break
    return traj

traj = probe_peakA("logpolar-r1", 6)
plt.figure(figsize=(6, 3))
plt.plot(traj, lw=1)
plt.axhline(1.0, color="r", ls="--", lw=1, label="|A|=1 stability bound")
plt.xlabel("optimizer step"); plt.ylabel("peak |A_bar|")
plt.title(f"peak |A| during QAT  (max over run = {max(traj):.5f})")
plt.legend(); plt.tight_layout(); plt.show()
print(f"peak |A| stays below 1: {max(traj) < 1.0}   (max = {max(traj):.6f})")
