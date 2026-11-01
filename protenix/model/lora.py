# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""LoRA (Low-Rank Adaptation) for finetuning the protenix trunk.

Wraps every `nn.Linear` (and subclass) descendant of a target trunk module
with a low-rank update branch W + B@A * (alpha/r), where A: (r, in),
B: (out, r). The base weight stays frozen; only A and B are trainable. This
keeps trunk knowledge intact while letting the model specialize to the
finetuning dataset with <1% extra trainable params and matching <1%
optimizer-state memory.

Usage:

    from protenix.model.lora import inject_lora
    if configs.lora.enabled:
        for mod_name in configs.lora.target_modules:
            inject_lora(
                getattr(model, mod_name),
                r=configs.lora.r,
                alpha=configs.lora.alpha,
                dropout=configs.lora.dropout,
            )

The base `Linear.forward` is preserved; LoRA delta is added on top:
    out = base(x) + (drop(x) @ A.T) @ B.T * (alpha / r)

Initialization: A ~ Kaiming uniform, B = 0 → LoRA contributes 0 at init, so the
finetuned model behaves identically to the base until training updates A/B.

LoRA params live under `<wrapper>.lora_A` / `lora_B`. The base linear's
weight/bias keep their original key paths (`<wrapper>.weight` / `<wrapper>.bias`),
so a pre-LoRA checkpoint can load with `strict=False` — only `lora_A` / `lora_B`
will be reported missing.

To save only LoRA deltas (portable adapters), use `lora_state_dict(model)`.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """Drop-in replacement for `nn.Linear` that adds a low-rank delta.

    The base linear's weight (and optional bias) are held as direct parameters
    of the wrapper — NOT as a `base` submodule — so the state_dict key path
    matches the original linear's path verbatim. A pre-LoRA checkpoint loads
    cleanly with `strict=False` (only the new `lora_A`, `lora_B` keys are
    reported missing); a saved LoRA checkpoint round-trips with `strict=True`.

    Output:
        y = F.linear(x, weight, bias) + dropout(x) @ A.T @ B.T * scaling

    where `scaling = alpha / r`. `lora_B` is zero-initialized so the wrapped
    module behaves identically to the base linear at training step 0.

    AMP / precision: if the source linear is a protenix `Linear` with a
    non-None `precision` attribute, we replay the same precision-cast path it
    used (autocast disabled, cast inputs/weight/bias to `precision`, cast back).
    For vanilla nn.Linear this branch is a no-op.

    Trainability: by default, only `lora_A` and `lora_B` have requires_grad=True.
    The base `weight` / `bias` inherit the requires_grad state of the source
    linear at injection time, so `_freeze_trunk` is respected.
    """

    def __init__(self, base_linear: nn.Linear, r: int=8, alpha: int=16, dropout: float=0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f'LoRA rank must be positive, got r={r}')
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        weight = base_linear.weight
        self.weight = nn.Parameter(weight.data, requires_grad=weight.requires_grad)
        if base_linear.bias is not None:
            bias = base_linear.bias
            self.bias = nn.Parameter(bias.data, requires_grad=bias.requires_grad)
        else:
            self.register_parameter('bias', None)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self._precision = getattr(base_linear, 'precision', None)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=weight.dtype, device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=weight.dtype, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def merged_weight(self) -> torch.Tensor:
        """Differentiable effective weight for fused kernels and export.

        Never cache or detach this tensor: fused kernels otherwise silently
        bypass the adapter. Merge in parameter precision, including under AMP.
        Input dropout cannot be represented by a single merged weight.
        """
        if self.training and getattr(self.lora_dropout, 'p', 0) > 0:
            raise RuntimeError('training LoRA dropout requires the unfused forward path')
        with torch.autocast(device_type=self.weight.device.type, enabled=False):
            return self.weight + self.lora_B @ self.lora_A * self.scaling

    def _base_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._precision is not None:
            input_dtype = x.dtype
            with torch.amp.autocast('cuda', enabled=False):
                b = self.bias.to(dtype=self._precision) if self.bias is not None else None
                return F.linear(x.to(dtype=self._precision), self.weight.to(dtype=self._precision), b).to(dtype=input_dtype)
        return F.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self._base_forward(x)
        lora_x = self.lora_dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling

def effective_linear_weight(layer: nn.Module) -> torch.Tensor:
    """Obtain weights without bypassing LoRA when a kernel owns the matmul."""
    return layer.merged_weight() if isinstance(layer, LoRALinear) else layer.weight

def inject_lora(root: nn.Module, r: int=8, alpha: int=16, dropout: float=0.0, verbose: bool=True, name_whitelist: 'set[str] | None'=None) -> int:
    """Walk `root`, replace every `nn.Linear` (or subclass) descendant with a
    `LoRALinear` wrapper.

    If `name_whitelist` is given (e.g. {"linear_k", "linear_v"}), only Linear
    layers whose attribute name on their parent is in the whitelist are
    wrapped — used by `lora.only_kv` to restrict LoRA to attention K/V
    projections.

    Returns the number of modules wrapped.

    Important caveats:
      - Call this AFTER `_freeze_trunk` (or any other base-param freeze). The
        wrapper inherits `base.requires_grad`, so frozen base params stay
        frozen; only `lora_A` / `lora_B` are trainable.
      - The wrapper holds base weight/bias as direct parameters (no `.base.`
        prefix in state_dict), so a pre-LoRA checkpoint loads cleanly with
        `strict=False` (only `lora_A` / `lora_B` are reported missing).
      - To put LoRA params in a separate optimizer group with `configs.finetune.lr`,
        pass `finetune_params_with_substring=["lora_A","lora_B"]`.
    """
    n_wrapped = 0
    to_replace: list[tuple[nn.Module, str, nn.Linear]] = []
    for module in root.modules():
        for attr_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if isinstance(child, LoRALinear):
                continue
            if name_whitelist is not None and attr_name not in name_whitelist:
                continue
            to_replace.append((module, attr_name, child))
    for parent, attr_name, child in to_replace:
        wrapped = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, wrapped)
        n_wrapped += 1
    if verbose:
        param_count = sum((p.numel() for n, p in root.named_parameters() if 'lora_A' in n or 'lora_B' in n))
        print(f'[LoRA] wrapped {n_wrapped} linear layers in {type(root).__name__}; added {param_count / 1000000.0:.3f}M trainable params (r={r}, alpha={alpha}, dropout={dropout})')
    return n_wrapped
