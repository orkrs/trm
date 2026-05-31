"""
TRM-Bank v3.0: Colab Evaluation Harness
========================================
Loads aifeifei798/Mamba3-MIMO-Tiny-HF weights into pure-PyTorch model.
"""

import math, os, re, subprocess, sys, time, warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

def _install() -> None:
    for pkg in [
        "torch>=2.4.0", "einops", "transformers>=4.44.0",
        "accelerate", "bitsandbytes", "safetensors", "huggingface_hub",
    ]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)

_install()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TRM-Bank] Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"[TRM-Bank] GPU: {torch.cuda.get_device_name(0)}", flush=True)

from transformers import AutoTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


# ========================================================================
# Pure-PyTorch Mamba3 with MIMO
# ========================================================================

class SimpleRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _rope(x: torch.Tensor, angles: torch.Tensor, half: int) -> torch.Tensor:
    x_ = rearrange(x[..., :half], "... (p two) -> ... p two", two=2)
    ca = angles[..., None].cos()
    sa = angles[..., None].sin()
    xr = torch.stack([
        x_[..., 0] * ca.squeeze(-1) - x_[..., 1] * sa.squeeze(-1),
        x_[..., 0] * sa.squeeze(-1) + x_[..., 1] * ca.squeeze(-1),
    ], dim=-1)
    xr = rearrange(xr, "... p two -> ... (p two)")
    return torch.cat([xr, x[..., half:]], dim=-1)


class PureMamba3Block(nn.Module):
    def __init__(self, d_model: int = 256, d_state: int = 64,
                 headdim: int = 64, mimo_rank: int = 2,
                 expand: int = 2, ngroups: int = 1,
                 rope_fraction: float = 0.5,
                 dt_min: float = 0.001, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4,
                 A_floor: float = 1e-4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.mimo_rank = mimo_rank
        self.expand = expand
        self.d_inner = expand * d_model
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = ngroups
        self.A_floor = A_floor

        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.split_tensor_size = split_tensor_size
        self.num_rope_angles = split_tensor_size // 2

        d_in_proj = (2 * self.d_inner +
                     2 * d_state * ngroups * mimo_rank +
                     3 * self.nheads +
                     self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        dt_init = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min))
        dt_init = torch.clamp(dt_init, min=dt_init_floor)
        dt_bias = dt_init + torch.log(-torch.expm1(-dt_init))
        self.dt_bias = nn.Parameter(dt_bias)

        self.B_bias = nn.Parameter(torch.zeros(self.nheads, mimo_rank, d_state) + 1.0)
        self.C_bias = nn.Parameter(torch.zeros(self.nheads, mimo_rank, d_state) + 1.0)
        self.B_norm = SimpleRMSNorm(d_state, eps=1e-5)
        self.C_norm = SimpleRMSNorm(d_state, eps=1e-5)

        self.mimo_x = nn.Parameter(torch.empty(self.nheads, mimo_rank, headdim))
        self.mimo_z = nn.Parameter(torch.empty(self.nheads, mimo_rank, headdim))
        self.mimo_o = nn.Parameter(torch.empty(self.nheads, mimo_rank, headdim))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in [self.mimo_x, self.mimo_z, self.mimo_o]:
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, _ = u.shape
        zxBCdt = self.in_proj(u)
        z, x_val, Bp, Cp, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.nheads, self.nheads, self.nheads,
             self.num_rope_angles],
            dim=-1)

        z = rearrange(z, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        x_val = rearrange(x_val, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        Bp = rearrange(Bp, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        Cp = rearrange(Cp, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)

        B_n = self.B_norm(Bp)
        C_n = self.C_norm(Cp)
        B_n_h = B_n.expand(-1, -1, -1, self.nheads, -1)
        C_n_h = C_n.expand(-1, -1, -1, self.nheads, -1)
        B_bias_a = rearrange(self.B_bias, "h r s -> 1 1 r h s")
        C_bias_a = rearrange(self.C_bias, "h r s -> 1 1 r h s")
        B_b = B_n_h + B_bias_a
        C_b = C_n_h + C_bias_a

        A_val = -F.softplus(dd_A)
        A_val = torch.clamp(A_val, max=-self.A_floor)
        DT_raw = F.softplus(dd_dt + self.dt_bias.unsqueeze(0).unsqueeze(0))
        ADT = A_val * DT_raw

        state = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                            device=u.device, dtype=x_val.dtype)
        outputs = []

        for t in range(L):
            adt_t = ADT[:, t, :]
            dt_t = DT_raw[:, t, :]
            x_t = x_val[:, t, :, :]
            z_t = z[:, t, :, :]
            B_t = B_b[:, t, :, :, :]
            C_t = C_b[:, t, :, :, :]
            ang_t = angles[:, t, :]

            hs = self.split_tensor_size

            B_flat = rearrange(B_t, "b r h s -> (b r h) s")
            C_flat = rearrange(C_t, "b r h s -> (b r h) s")
            ang_rha = rearrange(ang_t.unsqueeze(1).expand(-1, self.nheads, -1),
                                "b h a -> b 1 h a")
            ang_flat = rearrange(ang_rha.expand(-1, self.mimo_rank, -1, -1),
                                 "b r h a -> (b r h) a")
            B_rot = rearrange(
                _rope(B_flat, ang_flat, hs),
                "(b r h) s -> b r h s",
                b=B, r=self.mimo_rank, h=self.nheads)
            C_rot = rearrange(
                _rope(C_flat, ang_flat, hs),
                "(b r h) s -> b r h s",
                b=B, r=self.mimo_rank, h=self.nheads)

            decay = torch.exp(adt_t[:, :, None, None])
            decayed = state * decay

            x_proj = x_t.unsqueeze(2) * self.mimo_x.unsqueeze(0)
            z_proj = z_t.unsqueeze(2) * self.mimo_z.unsqueeze(0)

            total_inp = torch.zeros_like(state)
            for r in range(self.mimo_rank):
                x_r = x_proj[:, :, r, :]
                Br = B_rot[:, r, :, :]
                total_inp = total_inp + torch.einsum("bhp,bhs->bhps", x_r, Br)
            state = decayed + total_inp * dt_t[:, :, None, None]

            y_out = torch.zeros_like(x_t)
            for r in range(self.mimo_rank):
                z_r = z_proj[:, :, r, :]
                Cr = C_rot[:, r, :, :]
                y_r = torch.einsum("bhps,bhs->bhp", state, Cr)
                y_r = y_r * F.silu(z_r)
                y_out = y_out + y_r * self.mimo_o[None, :, r, :]

            y_out = y_out + self.D[None, :, None] * x_t
            outputs.append(y_out)

        y = torch.stack(outputs, dim=1)
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y)

    def forward_with_state(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = u.shape
        zxBCdt = self.in_proj(u)
        z, x_val, Bp, Cp, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.nheads, self.nheads, self.nheads,
             self.num_rope_angles],
            dim=-1)

        z = rearrange(z, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        x_val = rearrange(x_val, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        Bp = rearrange(Bp, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        Cp = rearrange(Cp, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)

        B_n = self.B_norm(Bp)
        C_n = self.C_norm(Cp)
        B_n_h = B_n.expand(-1, -1, -1, self.nheads, -1)
        C_n_h = C_n.expand(-1, -1, -1, self.nheads, -1)
        B_bias_a = rearrange(self.B_bias, "h r s -> 1 1 r h s")
        C_bias_a = rearrange(self.C_bias, "h r s -> 1 1 r h s")
        B_b = B_n_h + B_bias_a
        C_b = C_n_h + C_bias_a

        A_val = -F.softplus(dd_A)
        A_val = torch.clamp(A_val, max=-self.A_floor)
        DT_raw = F.softplus(dd_dt + self.dt_bias.unsqueeze(0).unsqueeze(0))
        ADT = A_val * DT_raw

        state = torch.zeros(B, self.nheads, self.headdim, self.d_state,
                            device=u.device, dtype=x_val.dtype)
        outputs = []

        for t in range(L):
            adt_t = ADT[:, t, :]
            dt_t = DT_raw[:, t, :]
            x_t = x_val[:, t, :, :]
            z_t = z[:, t, :, :]
            B_t = B_b[:, t, :, :, :]
            C_t = C_b[:, t, :, :, :]
            ang_t = angles[:, t, :]

            hs = self.split_tensor_size

            B_flat = rearrange(B_t, "b r h s -> (b r h) s")
            C_flat = rearrange(C_t, "b r h s -> (b r h) s")
            ang_rha = rearrange(ang_t.unsqueeze(1).expand(-1, self.nheads, -1),
                                "b h a -> b 1 h a")
            ang_flat = rearrange(ang_rha.expand(-1, self.mimo_rank, -1, -1),
                                 "b r h a -> (b r h) a")
            B_rot = rearrange(
                _rope(B_flat, ang_flat, hs),
                "(b r h) s -> b r h s",
                b=B, r=self.mimo_rank, h=self.nheads)
            C_rot = rearrange(
                _rope(C_flat, ang_flat, hs),
                "(b r h) s -> b r h s",
                b=B, r=self.mimo_rank, h=self.nheads)

            decay = torch.exp(adt_t[:, :, None, None])
            decayed = state * decay

            x_proj = x_t.unsqueeze(2) * self.mimo_x.unsqueeze(0)
            z_proj = z_t.unsqueeze(2) * self.mimo_z.unsqueeze(0)

            total_inp = torch.zeros_like(state)
            for r in range(self.mimo_rank):
                x_r = x_proj[:, :, r, :]
                Br = B_rot[:, r, :, :]
                total_inp = total_inp + torch.einsum("bhp,bhs->bhps", x_r, Br)
            state = decayed + total_inp * dt_t[:, :, None, None]

            y_out = torch.zeros_like(x_t)
            for r in range(self.mimo_rank):
                z_r = z_proj[:, :, r, :]
                Cr = C_rot[:, r, :, :]
                y_r = torch.einsum("bhps,bhs->bhp", state, Cr)
                y_r = y_r * F.silu(z_r)
                y_out = y_out + y_r * self.mimo_o[None, :, r, :]

            y_out = y_out + self.D[None, :, None] * x_t
            outputs.append(y_out)

        y = torch.stack(outputs, dim=1)
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y), state

    def allocate_inference_cache(self, batch_size: int) -> Dict[str, torch.Tensor]:
        dev = self.in_proj.weight.device
        return {
            "ssm_state": torch.zeros(batch_size, self.nheads, self.headdim,
                                     self.d_state, device=dev, dtype=torch.bfloat16),
        }

    def step(self, x_t: torch.Tensor,
             ssm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_t.shape[0]
        zxBCdt = self.in_proj(x_t)
        z, x_v, Bp, Cp, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdt,
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.d_state * self.num_bc_heads * self.mimo_rank,
             self.nheads, self.nheads, self.nheads,
             self.num_rope_angles],
            dim=-1)

        z = rearrange(z, "b (h p) -> b h p", h=self.nheads, p=self.headdim)
        x_v = rearrange(x_v, "b (h p) -> b h p", h=self.nheads, p=self.headdim)
        Bp = rearrange(Bp, "b (r g n) -> b r g n", r=self.mimo_rank, g=self.num_bc_heads)
        Cp = rearrange(Cp, "b (r g n) -> b r g n", r=self.mimo_rank, g=self.num_bc_heads)

        B_n = self.B_norm(Bp)
        C_n = self.C_norm(Cp)
        B_n_h = B_n.expand(-1, -1, self.nheads, -1)
        C_n_h = C_n.expand(-1, -1, self.nheads, -1)
        B_bias_a = rearrange(self.B_bias, "h r s -> 1 r h s")
        C_bias_a = rearrange(self.C_bias, "h r s -> 1 r h s")
        B_b = B_n_h + B_bias_a
        C_b = C_n_h + C_bias_a

        A_val = -F.softplus(dd_A.squeeze(1))
        A_val = torch.clamp(A_val, max=-self.A_floor)
        DT_r = F.softplus(dd_dt.squeeze(1) + self.dt_bias.unsqueeze(0))
        ADT = A_val * DT_r

        hs = self.split_tensor_size
        B_flat = rearrange(B_b, "b r h s -> (b r h) s")
        C_flat = rearrange(C_b, "b r h s -> (b r h) s")
        ang_rha = rearrange(angles.unsqueeze(1).expand(-1, self.nheads, -1),
                            "b h a -> b 1 h a")
        ang_flat = rearrange(ang_rha.expand(-1, self.mimo_rank, -1, -1),
                             "b r h a -> (b r h) a")
        B_rot = rearrange(
            _rope(B_flat, ang_flat, hs),
            "(b r h) s -> b r h s",
            b=B, r=self.mimo_rank, h=self.nheads)
        C_rot = rearrange(
            _rope(C_flat, ang_flat, hs),
            "(b r h) s -> b r h s",
            b=B, r=self.mimo_rank, h=self.nheads)

        decay = torch.exp(ADT[:, :, None, None])

        x_proj = x_v.unsqueeze(2) * self.mimo_x.unsqueeze(0)
        z_proj = z.unsqueeze(2) * self.mimo_z.unsqueeze(0)

        total_inp = torch.zeros_like(ssm_state)
        for r in range(self.mimo_rank):
            x_r = x_proj[:, :, r, :]
            Br = B_rot[:, r, :, :]
            total_inp = total_inp + torch.einsum("bhp,bhs->bhps", x_r, Br)
        ssm_state = ssm_state * decay + total_inp * DT_r[:, :, None, None]

        y_out = torch.zeros_like(x_v)
        for r in range(self.mimo_rank):
            z_r = z_proj[:, :, r, :]
            Cr = C_rot[:, r, :, :]
            y_r = torch.einsum("bhps,bhs->bhp", ssm_state, Cr)
            y_r = y_r * F.silu(z_r)
            y_out = y_out + y_r * self.mimo_o[None, :, r, :]

        y_out = y_out + self.D[None, :, None] * x_v
        y = rearrange(y_out, "b h p -> b (h p)")
        y = self.out_proj(y)
        return y, ssm_state.detach()


class PureMamba3Model(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 256,
                 n_layer: int = 2, d_state: int = 64,
                 headdim: int = 64, mimo_rank: int = 2,
                 chunk_size: int = 16) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            PureMamba3Block(d_model=d_model, d_state=d_state,
                            headdim=headdim, mimo_rank=mimo_rank)
            for _ in range(n_layer)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_f(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128,
                 temperature: float = 0.0, use_cache: bool = True) -> torch.Tensor:
        gen = input_ids.clone()

        if not use_cache:
            for _ in range(max_new_tokens):
                logits = self.forward(gen)
                nl = logits[:, -1, :]
                nt = nl.argmax(dim=-1, keepdim=True) if temperature == 0 else \
                     torch.multinomial(F.softmax(nl / temperature, dim=-1), 1)
                gen = torch.cat([gen, nt], dim=-1)
            return gen

        B = input_ids.shape[0]
        states: List[torch.Tensor] = []

        h = self.embedding(input_ids)
        for idx, layer in enumerate(self.layers):
            h, s = layer.forward_with_state(h)
            states.append(s)
        h = self.norm_f(h)
        logits = self.lm_head(h)

        for _ in range(max_new_tokens):
            nt = logits[:, -1].argmax(dim=-1, keepdim=True) if temperature == 0 else \
                 torch.multinomial(F.softmax(logits[:, -1] / temperature, dim=-1), 1)
            gen = torch.cat([gen, nt], dim=-1)
            x_t = self.embedding(nt).squeeze(1)
            for idx, layer in enumerate(self.layers):
                x_t, states[idx] = layer.step(x_t, states[idx])
            h_t = self.norm_f(x_t)
            logits = self.lm_head(h_t).unsqueeze(1)

        return gen


# ========================================================================
# Weight loading
# ========================================================================

def _download_weights() -> Dict[str, torch.Tensor]:
    repo = "aifeifei798/Mamba3-MIMO-Tiny-HF"
    subfolder = "mamba3_hf_ready"
    print(f"\n[TRM-Bank] Downloading weights from {repo}/{subfolder}...", flush=True)
    cache_path = hf_hub_download(repo_id=repo, filename="model.safetensors",
                                 subfolder=subfolder)
    print(f"[TRM-Bank] Loading safetensors from {cache_path}", flush=True)
    state = load_file(cache_path, device="cpu")
    print(f"[TRM-Bank] Loaded {len(state)} tensors", flush=True)
    for k, v in state.items():
        print(f"  {k}: {v.shape} {v.dtype}", flush=True)
    return state


def _build_from_state(state: Dict[str, torch.Tensor]) -> nn.Module:
    cfg = {
        "vocab_size": state["lm_head.weight"].shape[0],
        "d_model": state["embedding.weight"].shape[1],
        "d_state": state["layers.0.B_bias"].shape[-1],
        "headdim": state["layers.0.mimo_x"].shape[-1],
        "mimo_rank": state["layers.0.mimo_x"].shape[1],
    }
    n_layer = sum(1 for k in state if re.match(r"layers\.\d+\.in_proj\.weight", k))
    cfg["n_layer"] = n_layer
    print(f"[TRM-Bank] Config from state dict: {cfg}", flush=True)

    model = PureMamba3Model(**cfg)
    model.chunk_size = 1
    model.eval()

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[TRM-Bank] Missing keys: {missing}", flush=True)
    if unexpected:
        print(f"[TRM-Bank] Unexpected keys: {unexpected}", flush=True)
    if not missing and not unexpected:
        print(f"[TRM-Bank] All keys loaded successfully!", flush=True)

    model = model.to(device="cpu")
    model = model.bfloat16()
    model = model.to(device=device)

    total = sum(p.numel() for p in model.parameters())
    print(f"[TRM-Bank] Model has {total / 1e6:.1f}M params", flush=True)

    for name, p in model.named_parameters():
        if p.dtype != torch.bfloat16:
            print(f"  WARNING: {name} is {p.dtype}", flush=True)
    return model


def load_model() -> Tuple[nn.Module, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50256

    try:
        state = _download_weights()
        model = _build_from_state(state)
        chunk_size = getattr(model, "chunk_size", 16)
        print(f"[TRM-Bank] PureMamba3Model ready, chunk_size={chunk_size}", flush=True)
    except Exception as exc:
        print(f"[TRM-Bank] Weight loading failed: {exc}", flush=True)
        print("[TRM-Bank] Falling back to random weights...", flush=True)
        model = PureMamba3Model(
            vocab_size=50257, d_model=256, n_layer=2,
            d_state=64, headdim=64, mimo_rank=2,
        ).to(device).to(torch.bfloat16)
        model.eval()
        chunk_size = 1
        print(f"[TRM-Bank] Random-weight model ready", flush=True)

    return model, tokenizer, chunk_size


# ========================================================================
# Benchmarks
# ========================================================================

@dataclass
class ScoredResult:
    name: str
    prompt: str
    output: str
    latency_s: float
    vram_gb: float
    tokens_gen: int
    tokens_ps: float
    score: float
    verdict: str
    details: str


def get_vram() -> float:
    return torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0


def _extr_num(t: str) -> Optional[float]:
    for p in [r"answer is\s*(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)", r"result = (\d+(?:\.\d+)?)"]:
        m = re.search(p, t, re.I)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _extr_yn(t: str) -> Optional[str]:
    tl = t.lower()
    if re.search(r"\byes\b", tl):
        return "yes"
    if re.search(r"\bno\b", tl):
        return "no"
    return None


def _extr_grid(t: str) -> Optional[List[List[int]]]:
    g: List[List[int]] = []
    for l in t.strip().split("\n"):
        l = l.strip().strip("[](),.")
        if not l:
            continue
        try:
            r = [int(x) for x in re.findall(r"\d+", l)]
            if r:
                g.append(r)
        except ValueError:
            continue
    return g if g else None


def score_arc(o: str) -> Tuple[float, str]:
    g = _extr_grid(o)
    if g is None:
        return 0.0, "no grid"
    t = [[0, 1], [1, 0]]
    if len(g) != 2 or len(g[0]) != 2:
        return 0.2, f"shape mismatch: {g}"
    return (1.0, f"correct: {g}") if g == t else (0.3, f"wrong: {g} vs {t}")


def score_gsm(o: str) -> Tuple[float, str]:
    n = _extr_num(o)
    if n is None:
        return 0.0, "no number"
    if abs(n - 7.0) < 0.01:
        return 1.0, f"correct: {n}"
    return 0.2, f"got {n}, expected 7"


def score_hum(e: str) -> Tuple[float, str]:
    s = 0.0
    notes = []
    if "def add_vectors" in e:
        s += 0.4
        notes.append("has def")
    if "return" in e:
        s += 0.3
        notes.append("has return")
    if "zip(" in e or "range(" in e or "len(" in e:
        s += 0.3
        notes.append("has iteration")
    return round(s, 2), ", ".join(notes) if notes else "no pattern"


def score_sud(o: str) -> Tuple[float, str]:
    g = _extr_grid(o)
    nums = [int(x) for x in re.findall(r"\d+", o) if 0 <= int(x) <= 9]
    flat = [v for row in g for v in row] if g else nums
    if len(flat) < 4:
        return 0.0, f"not enough: {flat}"
    v = flat[:4]
    r1, r2 = v[0] + v[1], v[2] + v[3]
    c1, c2 = v[0] + v[2], v[1] + v[3]
    s = sum([0.25 for x in [r1, r2, c1, c2] if x == 3])
    notes = [f"row0 sum={r1}", f"row1 sum={r2}", f"col0 sum={c1}", f"col1 sum={c2}"]
    return (1.0, "perfect") if v == [1, 2, 1, 2] else (round(s, 2), ", ".join(notes))


def score_pro(o: str) -> Tuple[float, str]:
    a = _extr_yn(o)
    if a is None:
        return 0.0, "no yes/no"
    return (1.0, "correct: yes") if a == "yes" else (0.0, f"got '{a}'")


SCORERS: Dict[str, Callable[[str], Tuple[float, str]]] = {
    "ARC-AGI": score_arc, "GSM8K": score_gsm, "HumanEval": score_hum,
    "Sudoku-Extreme": score_sud, "ProntoQA": score_pro,
}

BENCHMARKS: List[Tuple[str, str, int]] = [
    ("ARC-AGI", "Input grid: [[1, 0], [0, 1]]. "
     "Transform this grid by flipping it horizontally. Output DSL commands:", 64),
    ("GSM8K", "Question: John has 5 apples. He buys 2 more. "
     "How many apples does he have? Write a Python equation to solve this:", 64),
    ("HumanEval", "Write a Python function `def add_vectors(a, b):` "
     "that adds two lists element-wise:", 96),
    ("Sudoku-Extreme", "Grid: [[0, 2], [1, 0]]. "
     "Constraints: rows must sum to 3, columns must sum to 3. Solve the matrix:", 64),
    ("ProntoQA", "Premise 1: Every cat is an animal. "
     "Premise 2: Fae is a cat. Question: Is Fae an animal? Deduce step-by-step:", 96),
]


def _pad_to_multiple(t: torch.Tensor, multiple: int, pad_val: int = 0) -> Tuple[torch.Tensor, int]:
    length = t.shape[1]
    if length % multiple == 0:
        return t, length
    pad_len = multiple - length % multiple
    return F.pad(t, (0, pad_len), value=pad_val), length


def run_bench(model: nn.Module, tok: Any, name: str, prompt: str,
              max_new: int = 64, temp: float = 0.0,
              chunk_size: int = 1) -> ScoredResult:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    inputs = tok(prompt, return_tensors="pt")
    ids = inputs.input_ids.to(device)
    pad_id = getattr(tok, "pad_token_id", 0) or 0
    in_len = ids.shape[1]
    needs_pad = chunk_size > 1

    with torch.no_grad():
        if needs_pad:
            out_ids = ids.clone()
            for _ in range(max_new):
                x_pad, orig_len = _pad_to_multiple(out_ids, chunk_size, pad_id)
                log = model(x_pad)
                nl = log[:, orig_len - 1, :]
                nt = nl.argmax(dim=-1, keepdim=True) if temp == 0 else \
                     torch.multinomial(F.softmax(nl / temp, dim=-1), 1)
                out_ids = torch.cat([out_ids, nt], dim=-1)
        elif hasattr(model, "generate"):
            out_ids = model.generate(ids, max_new_tokens=max_new, temperature=temp, use_cache=True)
        else:
            out_ids = ids.clone()
            for _ in range(max_new):
                log = model(out_ids)
                nl = log[:, -1, :]
                nt = nl.argmax(dim=-1, keepdim=True) if temp == 0 else \
                     torch.multinomial(F.softmax(nl / temp, dim=-1), 1)
                out_ids = torch.cat([out_ids, nt], dim=-1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    el = time.time() - t0
    vram_p = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    out_len = out_ids.shape[1] - in_len
    tps = out_len / el if el > 0 else 0.0
    text = tok.decode(out_ids[0, in_len:].tolist(), skip_special_tokens=True)

    scorer = SCORERS.get(name)
    sc, det = scorer(text) if scorer else (0.0, "no scorer")
    verdict = "PASS" if sc >= 0.8 else ("PARTIAL" if sc >= 0.3 else "FAIL")
    sym = {"PASS": "\u2713", "FAIL": "\u2717", "PARTIAL": "~"}[verdict]

    r = ScoredResult(name, prompt, text, round(el, 3), round(vram_p, 3),
                     out_len, round(tps, 1), round(sc, 2), verdict, det)
    print(f"\n{'=' * 62}\n  {sym}  {name:<20}  Score: {sc:.0%}  {verdict}\n"
          f"  {'-' * 58}\n  Prompt : {prompt[:70]}...\n"
          f"  Output : {text[:80]}\n  Details: {det}\n"
          f"  Latency: {el:.3f}s  |  {out_len} tok ({tps:.1f} tok/s)"
          f"  |  VRAM: {vram_p:.2f} GB\n{'=' * 62}\n", flush=True)
    return r


def main() -> None:
    print(f"\n{'#' * 62}\n#  TRM-Bank v3.0: Full Evaluation Suite\n"
          f"#  Device: {device}\n{'#' * 62}\n", flush=True)

    model, tokenizer, chunk_size = load_model()
    print(f"\n[TRM-Bank] Running {len(BENCHMARKS)} benchmarks...\n"
          f"[TRM-Bank] chunk_size={chunk_size}\n", flush=True)

    results = [run_bench(model, tokenizer, n, p, m, chunk_size=chunk_size)
               for n, p, m in BENCHMARKS]

    passed = sum(1 for r in results if r.verdict == "PASS")
    partial = sum(1 for r in results if r.verdict == "PARTIAL")
    failed = sum(1 for r in results if r.verdict == "FAIL")
    avg_sc = sum(r.score for r in results) / len(results)

    print(f"\n{'#' * 62}\n#  FINAL SCOREBOARD\n{'#' * 62}", flush=True)
    hdr = f"{'Benchmark':<20} {'Score':<8} {'Verdict':<10} {'Latency':<10} {'tok/s':<8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    tl, tt = 0.0, 0
    for r in results:
        sym = {"PASS": "\u2713", "FAIL": "\u2717", "PARTIAL": "~"}[r.verdict]
        print(f"{r.name:<20} {r.score:<8.0%} {r.verdict:<10} "
              f"{r.latency_s:<10.3f} {r.tokens_ps:<8.1f}", flush=True)
        tl += r.latency_s
        tt += r.tokens_gen
    print("-" * len(hdr), flush=True)
    print(f"{'AVERAGE':<20} {avg_sc:<8.0%} {passed}/{len(results)} pass   "
          f"{tl:<10.3f} {tt / max(0.001, tl):<8.1f}", flush=True)
    print(f"\n  Passed: {passed}  |  Partial: {partial}  |  Failed: {failed}\n", flush=True)
    print(f"[TRM-Bank] Evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
