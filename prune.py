#!/usr/bin/env python3
"""
Deep Compression for MobileNetV2 on CIFAR-10
Based on "Deep Compression" (Han et al., ICLR 2016)
Three-stage pipeline: Pruning → Quantization → Huffman Coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import Dict, Tuple

# Import your training utilities
from MobilenetV2_train import (
    MobileNetV2, MV2Config,
    build_cifar10_loaders,
    evaluate
)
import torch
import torch.nn as nn
from contextlib import contextmanager

# -------- Post-Training Activation Quantizer --------
class ActivationPTQ(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = True, ema_decay: float = 0.9):
        super().__init__()
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.ema_decay = ema_decay
        self.register_buffer("min_val", torch.tensor(float("+inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale",   torch.tensor(1.0))
        self.register_buffer("zp",      torch.tensor(0.0))
        self.register_buffer("calibrating", torch.tensor(1, dtype=torch.uint8))
        self.register_buffer("frozen",      torch.tensor(0, dtype=torch.uint8))
        self.last_shape = None
        self.last_numel = 0

    def _ensure_device(self, x: torch.Tensor):
        dev = x.device
        if self.min_val.device != dev:
            self.min_val = self.min_val.to(dev)
            self.max_val = self.max_val.to(dev)
            self.scale   = self.scale.to(dev)
            self.zp      = self.zp.to(dev)
            self.calibrating = self.calibrating.to(dev)
            self.frozen      = self.frozen.to(dev)

    def _update_range(self, x: torch.Tensor):
        self._ensure_device(x)
        d = self.ema_decay
        bmin = x.min().detach().to(self.min_val.dtype)
        bmax = x.max().detach().to(self.max_val.dtype)
        if torch.isinf(self.min_val):
            self.min_val.copy_(bmin)
            self.max_val.copy_(bmax)
        else:
            self.min_val.mul_(d).add_(bmin * (1 - d))
            self.max_val.mul_(d).add_(bmax * (1 - d))

    @torch.no_grad()
    def set_calib(self, enable: bool):
        """Set calibration mode and freeze scales when done."""
        self.calibrating.fill_(1 if enable else 0)
        if not enable and int(self.frozen.item()) == 0:
            mn = float(self.min_val.item())
            mx = float(self.max_val.item())
            if mx <= mn:
                mx, mn = (1.0, 0.0) if not self.symmetric else (1.0, -1.0)
            if self.symmetric:
                qmin, qmax = -(2**(self.n_bits-1)), (2**(self.n_bits-1)) - 1
                a = max(abs(mn), abs(mx))
                self.scale.fill_(a / max(qmax, 1))
                self.zp.fill_(0.0)
            else:
                qmin, qmax = 0, (2**self.n_bits) - 1
                self.scale.fill_((mx - mn) / max((qmax - qmin), 1))
                self.zp.fill_(qmin - mn / (self.scale.item() + 1e-12))
            self.frozen.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Track shape for memory estimation
        self.last_shape = tuple(x.shape)
        self.last_numel = x.numel()
        
        if int(self.calibrating.item()) == 1:
            self._update_range(x)
            return x
        if int(self.frozen.item()) == 0:
            return x
        
        self._ensure_device(x)
        qmin, qmax = (-(2**(self.n_bits-1)), (2**(self.n_bits-1)) - 1) if self.symmetric else (0, (2**self.n_bits) - 1)
        s = self.scale
        zp = torch.round(self.zp) if not self.symmetric else 0.0
        x_q = torch.clamp(torch.round(x / (s + 1e-12) + zp), qmin, qmax)
        return (x_q - zp) * s
# ---- utilities: insert quantizers after activations, calibrate, then freeze ----
_ACT_CLASSES = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.LeakyReLU)

def _attach_act_quantizers(model: nn.Module, n_bits: int = 8, symmetric: bool = True):
    """
    Replace each activation module A with nn.Sequential(A, ActivationPTQ(...)).
    """
    for name, m in list(model.named_children()):
        if isinstance(m, _ACT_CLASSES):
            setattr(model, name, nn.Sequential(m, ActivationPTQ(n_bits=n_bits, symmetric=symmetric)))
        else:
            _attach_act_quantizers(m, n_bits, symmetric)  # recurse

def _iter_act_quantizers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, ActivationPTQ):
            yield m

@torch.no_grad()
def calibrate_activations(model: nn.Module, loader, device: torch.device, num_batches: int = 20):
    """
    Run a few batches to collect activation ranges (no quantization applied during this pass).
    """
    model.eval()
    for q in _iter_act_quantizers(model):
        q.set_calib(True)

    it = iter(loader)
    for _ in range(max(1, num_batches)):
        try:
            x, _y = next(it)
        except StopIteration:
            it = iter(loader)
            x, _y = next(it)
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            _ = model(x)

    # freeze scales and enable fake-quant for inference
    for q in _iter_act_quantizers(model):
        q.set_calib(False)

@contextmanager
def temporarily_disable_act_quant(model: nn.Module):
    """
    Context manager to run without fake-quant (e.g., if you need a clean pass).
    """
    prev = []
    for q in _iter_act_quantizers(model):
        prev.append(int(q.calibrating.item()))
        q.set_calib(True)   # treat as pass-through
    try:
        yield
    finally:
        # restore to frozen/fake-quant mode
        for q in _iter_act_quantizers(model):
            q.set_calib(False)


# ============================================================================
# STAGE 1: PRUNING (Magnitude-based)
# ============================================================================

def prune_network(model: nn.Module, sparsity: float = 0.5, exclude_first_last: bool = True) -> Dict:
    print(f"\n{'='*60}\nSTAGE 1: PRUNING\n{'='*60}")
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    if exclude_first_last and len(layers) >= 2:
        layers = layers[1:-1]

    with torch.no_grad():
        # gather on CPU to avoid GPU spikes
        all_w = torch.cat([m.weight.detach().abs().flatten().cpu() for _, m in layers], dim=0)
        N = all_w.numel()
        # kthvalue is 1-indexed; clamp to [1, N]
        k = max(1, min(int(N * sparsity), N))
        th = torch.kthvalue(all_w, k).values.to(next(model.parameters()).device)

        masks, total, pruned = {}, 0, 0
        for name, m in layers:
            w = m.weight.data
            mask = (w.abs() >= th).to(w.dtype)
            m.weight.data.mul_(mask)
            masks[name] = mask
            total += w.numel()
            pruned += int((mask == 0).sum())
    s = pruned / max(1, total)
    print(f"Pruned {pruned:,}/{total:,} params | Sparsity {s*100:.2f}% | Dense-compression {1/(1-s):.2f}x")
    return masks

# ============================================================================
# STAGE 2: TRAINED QUANTIZATION (Weight Sharing via K-means)
# ============================================================================

def linear_initialization(weight: torch.Tensor, n_clusters: int) -> torch.Tensor:
    wmin = weight.min().item()
    wmax = weight.max().item()
    if wmin == wmax:
        return torch.linspace(wmin - 1e-6, wmax + 1e-6, n_clusters, device=weight.device)
    return torch.linspace(wmin, wmax, n_clusters, device=weight.device)

def _nearest_centroid_ids(x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    # x: [N], C: [K]
    # dist^2 = (x - c)^2 = x^2 + c^2 - 2xc
    x2 = x.pow(2).unsqueeze(1)          # [N,1]
    c2 = C.pow(2).unsqueeze(0)          # [1,K]
    d2 = x2 + c2 - 2.0 * x.unsqueeze(1) * C.unsqueeze(0)  # [N,K]
    return d2.argmin(dim=1)             # [N]

def quantize_layer(weight: torch.Tensor, mask: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    K = 2 ** n_bits
    nz_mask = mask.bool()
    if nz_mask.sum().item() == 0:
        # No nonzeros — keep as-is; codebook with 1 entry to avoid degenerate Huffman later
        return weight, torch.zeros(max(1, K), device=weight.device)

    x = weight[nz_mask]
    C = linear_initialization(x, K)

    # Lloyd’s k-means
    for _ in range(50):
        ids = _nearest_centroid_ids(x, C)
        newC = C.clone()
        for k in range(K):
            sel = (ids == k)
            if sel.any():
                newC[k] = x[sel].mean()
        if torch.allclose(C, newC, rtol=1e-4, atol=1e-6):
            C = newC
            break
        C = newC

    # Quantize
    q = weight.clone()
    ids_full = _nearest_centroid_ids(weight[nz_mask], C)
    q[nz_mask] = C[ids_full]
    return q, C

def quantize_network(model: nn.Module, masks: Dict, 
                     conv_bits: int = 8, fc_bits: int = 5) -> Dict:
    """
    Quantize all layers in the network.
    Conv layers: 8 bits (256 clusters)
    FC layers: 5 bits (32 clusters)
    """
    print(f"\n{'='*60}")
    print("STAGE 2: QUANTIZATION")
    print(f"{'='*60}")
    
    codebooks = {}
    
    for name, m in model.named_modules():
        if name not in masks:
            continue
            
        n_bits = fc_bits if isinstance(m, nn.Linear) else conv_bits
        
        quantized_weight, codebook = quantize_layer(
            m.weight.data, masks[name], n_bits
        )
        
        m.weight.data = quantized_weight
        codebooks[name] = codebook
        
        print(f"{name}: {n_bits} bits ({2**n_bits} clusters)")
    
    return codebooks


# ========= Sparse (CSR) RAM Size Estimator ==================================
def _ceil_log2(n: int) -> int:
    return 0 if n <= 1 else math.ceil(math.log2(n))

def _conv_rows_cols(m: nn.Conv2d) -> tuple[int, int]:
    # CSR view for conv weights: rows=out_channels, cols=in_channels * kH * kW
    kH, kW = m.kernel_size
    rows = m.out_channels
    cols = m.in_channels * kH * kW
    return rows, cols

def estimate_ram_mb_csr(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    codebooks: Dict[str, torch.Tensor],
    *,
    centroid_bits: int = 16,     # fp16 centroids by default
    rowptr_bits: int = 32,       # 32-bit row pointers
    colind_bits_auto16: bool = True,  # use 16-bit col indices when cols<=65535, else 32
    colind_bits_override: int | None = None,  # set to 8/16/32/64 to force a width
    bias_on_active_rows: bool = True,
    extra_overhead_mb: float = 5.0,    # your original fixed overhead
) -> float:
    """
    Returns RAM in MB for CSR sparse + codebook indices (no Huffman).
    Counts:
      - values (indices): nnz * ceil(log2(K)) bits
      - CSR colind: nnz * (16 or 32) bits (auto) unless override provided
      - CSR rowptr: (rows+1) * rowptr_bits
      - codebook: K * centroid_bits
      - bias: (#active_rows or rows) * centroid_bits (if bias exists)
    """
    total_bits = 0

    for name, module in model.named_modules():
        if name not in masks or name not in codebooks:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        mask = masks[name].to(torch.bool)
        K = int(codebooks[name].numel())
        if K <= 1:
            k_bits = 0
        else:
            k_bits = _ceil_log2(K)

        # nnz and matrix dims
        nnz = int(mask.sum().item())
        if nnz == 0:
            # still pay codebook, and possibly bias if you keep it
            codebook_bits = K * centroid_bits
            bias_bits = 0
            if module.bias is not None and not bias_on_active_rows:
                rows = module.out_features if isinstance(module, nn.Linear) else module.out_channels
                bias_bits = rows * centroid_bits
            total_bits += codebook_bits + bias_bits
            continue

        if isinstance(module, nn.Linear):
            rows, cols = module.out_features, module.in_features
            active_rows = int(mask.any(dim=1).sum().item())
        else:
            rows, cols = _conv_rows_cols(module)
            active_rows = int(mask.view(mask.shape[0], -1).any(dim=1).sum().item())

        # values (indices into codebook)
        values_bits = nnz * k_bits

        # CSR column index width
        if colind_bits_override is not None:
            ci_bits = int(colind_bits_override)
            assert ci_bits in (8, 16, 32, 64)
        else:
            if colind_bits_auto16 and cols <= 65535:
                ci_bits = 16
            else:
                ci_bits = 32

        colind_bits = nnz * ci_bits
        rowptr_bits_total = (rows + 1) * rowptr_bits

        # codebook storage (centroids)
        codebook_bits = K * centroid_bits

        # bias storage
        bias_bits = 0
        if module.bias is not None:
            n_bias = active_rows if bias_on_active_rows else rows
            bias_bits = n_bias * centroid_bits

        layer_bits = values_bits + colind_bits + rowptr_bits_total + codebook_bits + bias_bits
        total_bits += layer_bits

    ram_mb = total_bits / (8 * 1024 * 1024) + float(extra_overhead_mb)
    return ram_mb


def estimate_ram_mb_csr_verbose(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    codebooks: Dict[str, torch.Tensor],
    *,
    centroid_bits: int = 16,
    rowptr_bits: int = 32,
    colind_bits_auto16: bool = True,
    colind_bits_override: int | None = None,
    bias_on_active_rows: bool = True,
    extra_overhead_mb: float = 5.0,
) -> float:
    total_bits = 0

    print(f"\n{'='*80}")
    print("RAM ESTIMATION — CSR sparse + codebook indices (no Huffman)")
    print(f"{'='*80}")

    for name, module in model.named_modules():
        if name not in masks or name not in codebooks:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        mask = masks[name].to(torch.bool)
        weight = module.weight
        K = int(codebooks[name].numel())
        k_bits = 0 if K <= 1 else _ceil_log2(K)

        nnz = int(mask.sum().item())
        if isinstance(module, nn.Linear):
            rows, cols = module.out_features, module.in_features
            active_rows = int(mask.any(dim=1).sum().item())
        else:
            rows, cols = _conv_rows_cols(module)
            active_rows = int(mask.view(mask.shape[0], -1).any(dim=1).sum().item())

        if colind_bits_override is not None:
            ci_bits = int(colind_bits_override)
            assert ci_bits in (8,16,32,64)
        else:
            ci_bits = 16 if (colind_bits_auto16 and cols <= 65535) else 32

        values_bits = nnz * k_bits
        colind_bits = nnz * ci_bits
        rowptr_bits_total = (rows + 1) * rowptr_bits
        codebook_bits = K * centroid_bits
        bias_bits = 0
        if module.bias is not None:
            n_bias = active_rows if bias_on_active_rows else rows
            bias_bits = n_bias * centroid_bits

        layer_bits = values_bits + colind_bits + rowptr_bits_total + codebook_bits + bias_bits
        total_bits += layer_bits

        print(f"\n{name}:")
        print(f"  Shape: rows={rows}, cols={cols} | nnz={nnz:,} | active_rows={active_rows}")
        print(f"  Codebook:  K={K}, centroids={codebook_bits/8/1024:.2f} KB  ({centroid_bits} bits each)")
        print(f"  Values:    {values_bits/8/1024:.2f} KB  ({nnz:,} × {k_bits} bits index)")
        print(f"  colind:    {colind_bits/8/1024:.2f} KB  ({nnz:,} × {ci_bits}-bit col index)")
        print(f"  rowptr:    {rowptr_bits_total/8/1024:.2f} KB  ({rows+1} × {rowptr_bits}-bit)")
        if bias_bits:
            print(f"  Bias:      {bias_bits/8/1024:.2f} KB  ({'active rows' if bias_on_active_rows else 'all rows'})")
        print(f"  TOTAL:     {layer_bits/8/1024:.2f} KB")

    data_mb = total_bits / (8 * 1024 * 1024)
    total_mb = data_mb + float(extra_overhead_mb)

    print(f"\n{'='*80}")
    print(f"Model data:   {data_mb:>10.3f} MB")
    print(f"Overhead:     {extra_overhead_mb:>10.3f} MB")
    print(f"Total RAM:    {total_mb:>10.3f} MB")
    print(f"{'='*80}\n")

    # Optional comparison with dense FP32 params
    original_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
    print(f"Original (FP32):  {original_mb:.2f} MB")
    print(f"Compressed:       {total_mb:.2f} MB")
    print(f"Compression:      {original_mb/total_mb:.2f}x")
    print(f"Saved:            {original_mb-total_mb:.2f} MB ({(original_mb-total_mb)/original_mb*100:.1f}%)\n")

    return total_mb

def estimate_ram_mb(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    codebooks: Dict[str, torch.Tensor]
) -> float:
    """
    Returns RAM in MB. Uses minimum bits for everything:
    - Codebook: ceil(log2(K)) bits per value
    - Indices: ceil(log2(K)) bits per non-zero weight
    - Positions: ceil(log2(total_weights)) bits per non-zero weight
    - Bias: 16 bits (fp16)
    """
    
    total_bits = 0
    
    for name, module in model.named_modules():
        if name not in masks or name not in codebooks:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        
        mask = masks[name]
        codebook = codebooks[name]
        weight = module.weight
        
        # Count non-zero weights
        nnz = int(mask.sum().item())
        if nnz == 0:
            continue
        
        # Codebook size: K values, each needing ceil(log2(K)) bits minimum
        # But we still store them as fp16 (16 bits each)
        K = len(codebook)
        codebook_bits = K * 16  # fp16 storage
        
        # Index bits: ceil(log2(K)) bits per non-zero weight
        bits_per_index = max(1, math.ceil(math.log2(K))) if K > 1 else 1
        indices_bits = nnz * bits_per_index
        
        # Position bits: ceil(log2(total_weights)) bits per non-zero weight
        total_weights = weight.numel()
        bits_per_position = max(1, math.ceil(math.log2(total_weights)))
        positions_bits = nnz * bits_per_position
        
        # Bias: fp16 (16 bits per value)
        bias_bits = 0
        if module.bias is not None:
            bias_bits = module.bias.numel() * 16
        
        total_bits += codebook_bits + indices_bits + positions_bits + bias_bits
    
    # Convert bits to MB and add 5MB overhead
    ram_mb = total_bits / (8 * 1024 * 1024) + 5.0
    
    return ram_mb

''''''''''''''''''
def model_size_bytes_fp32(model):
    """Total size of all parameters if stored as FP32 (4 bytes each)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * 4
    return total

def model_size_bytes_quant(model, weight_bits=8):
    """Total size if all weights were stored as intN, biases stay FP32."""
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total += p.numel() * weight_bits // 8  # intN
        elif "bias" in name:
            total += p.numel() * 4                 # keep biases FP32
    return total



def estimate_ram_mb_verbose(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    codebooks: Dict[str, torch.Tensor]
) -> float:
    """
    Same as estimate_ram_mb but prints details.
    """
    
    total_bits = 0
    
    print(f"\n{'='*80}")
    print("RAM ESTIMATION (Minimum Bits)")
    print(f"{'='*80}")
    
    for name, module in model.named_modules():
        if name not in masks or name not in codebooks:
            continue
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        
        mask = masks[name]
        codebook = codebooks[name]
        weight = module.weight
        
        # Count non-zero weights
        nnz = int(mask.sum().item())
        if nnz == 0:
            continue
        
        K = len(codebook)
        total_weights = weight.numel()
        
        # Codebook: K values in fp16
        codebook_bits = K * 16
        
        # Indices: ceil(log2(K)) bits per weight
        bits_per_index = max(1, math.ceil(math.log2(K))) if K > 1 else 1
        indices_bits = nnz * bits_per_index
        
        # Positions: ceil(log2(total_weights)) bits per weight
        bits_per_position = max(1, math.ceil(math.log2(total_weights)))
        positions_bits = nnz * bits_per_position
        
        # Bias: fp16
        bias_bits = 0
        if module.bias is not None:
            bias_bits = module.bias.numel() * 16
        
        layer_bits = codebook_bits + indices_bits + positions_bits + bias_bits
        total_bits += layer_bits
        
        print(f"\n{name}:")
        print(f"  Non-zeros: {nnz:,} / {total_weights:,} ({nnz/total_weights*100:.1f}%)")
        print(f"  Codebook:  {codebook_bits/8/1024:.2f} KB  (K={K}, 16 bits each)")
        print(f"  Indices:   {indices_bits/8/1024:.2f} KB  ({nnz:,} × {bits_per_index} bits)")
        print(f"  Positions: {positions_bits/8/1024:.2f} KB  ({nnz:,} × {bits_per_position} bits)")
        if bias_bits > 0:
            print(f"  Bias:      {bias_bits/8/1024:.2f} KB  ({module.bias.numel()} × 16 bits)")
        print(f"  Total:     {layer_bits/8/1024:.2f} KB")
    
    # Totals
    data_mb = total_bits / (8 * 1024 * 1024)
    overhead_mb = 5.0
    total_mb = data_mb + overhead_mb
    
    print(f"\n{'='*80}")
    print(f"Model data:   {data_mb:>10.3f} MB")
    print(f"Overhead:     {overhead_mb:>10.3f} MB")
    print(f"Total RAM:    {total_mb:>10.3f} MB")
    print(f"{'='*80}\n")
    
    # Compare with original
    original_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)
    print(f"Original (FP32):  {original_mb:.2f} MB")
    print(f"Compressed:       {total_mb:.2f} MB")
    print(f"Compression:      {original_mb/total_mb:.2f}x")
    print(f"Saved:            {original_mb-total_mb:.2f} MB ({(original_mb-total_mb)/original_mb*100:.1f}%)\n")
    
    return total_mb
@torch.no_grad()
def orig_activation_bits(model: nn.Module, input_shape=(1,3,32,32), act_bits=32) -> int:
    dev = next(model.parameters()).device
    was_train = model.training
    model.eval()
    total = 0
    hooks = []
    def hook(_m,_i,o):
        nonlocal total
        if torch.is_tensor(o): total += o.numel() * act_bits
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.LeakyReLU)):
            hooks.append(m.register_forward_hook(hook))
    _ = model(torch.empty(*input_shape, device=dev))
    for h in hooks: h.remove()
    if was_train: model.train()
    return total
# ============================================================================
# ============================================================================
# HAWQ-style Mixed-Precision Quantization (Alternative to uniform quantization)
# ============================================================================

@torch.no_grad()
def _flatten_like(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)

def _normalize_(v: torch.Tensor):
    v /= (v.norm() + 1e-12)
    return v

def _layer_params(model: nn.Module, masks: Dict) -> list:
    """Return [(name, module)] only for layers we will quantize (keys in masks)."""
    layers = []
    for name, m in model.named_modules():
        if name in masks and isinstance(m, (nn.Conv2d, nn.Linear)):
            layers.append((name, m))
    return layers

def _top_hessian_eig_for_layer(
    model: nn.Module,
    layer_module: nn.Module,
    mask: torch.Tensor,          # {0,1} same shape as weight
    loss_fn,
    data_iter,                   # an iterator or a DataLoader
    device,
    power_iters: int = 8,
    batches: int = 2,
) -> float:
    """
    Mask-aware power iteration for the top block-Hessian eigenvalue.
    Restricts everything to the unpruned subspace defined by `mask`.
    """
    import math
    W = layer_module.weight
    # --- ensure mask matches device & dtype ---
    mask_f = mask.to(device=device, dtype=W.dtype)
    mask_b = mask_f.bool()

    # helper: masked L2 norm
    def mnorm(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(((x * mask_f) ** 2).sum() + 1e-12)

    # --- init v on masked subspace ---
    v = torch.zeros_like(W, device=device)
    nnz = int(mask_b.sum().item())
    if nnz == 0:
        return 0.0
    v[mask_b] = torch.randn(nnz, device=device)
    v = (v * mask_f) / mnorm(v)

    # make data_iter cycle
    if isinstance(data_iter, torch.utils.data.DataLoader):
        data_iter = iter(data_iter)

    def next_batch(it):
        nonlocal data_iter
        try:
            x, y = next(it)
        except StopIteration:
            data_iter = iter(data_iter)
            x, y = next(data_iter)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    ests = []
    for _ in range(batches):
        x, y = next_batch(data_iter)

        prev_rq = None
        stuck = 0
        for _ in range(power_iters):
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)

            # g = dL/dW (need graph for HVP)
            g = torch.autograd.grad(loss, W, create_graph=True, retain_graph=True)[0]

            v_m = v * mask_f
            gv = (g * v_m).sum()

            # Hv = d(g·v)/dW, then project to masked subspace
            Hv = torch.autograd.grad(gv, W, retain_graph=False, create_graph=False)[0]
            Hv = Hv * mask_f

            # Rayleigh quotient on masked subspace
            num = (v_m * Hv).sum()
            den = (v_m ** 2).sum() + 1e-12
            rq = (num / den).abs().item()
            ests.append(rq)

            # Next iterate (masked + normalized)
            v = Hv.detach()
            nrm = mnorm(v)
            if nrm.item() < 1e-12:
                # re-init on mask if we hit the null space
                v.zero_()
                v[mask_b] = torch.randn(nnz, device=device)
                v = (v * mask_f) / mnorm(v)
            else:
                v = v / nrm

            # early stop if stabilized
            if prev_rq is not None and abs(rq - prev_rq) < 1e-3:
                stuck += 1
                if stuck >= 2:
                    break
            else:
                stuck = 0
            prev_rq = rq

    if not ests:
        return 0.0
    ests.sort()
    mid = len(ests) // 2
    return float(ests[mid] if len(ests) % 2 else 0.5 * (ests[mid - 1] + ests[mid]))


def _assign_bits_by_sensitivity(
    sens: dict,
    layers_ordered: list,
    bit_bins: tuple = (8, 6, 4, 3, 2),
    proportions: tuple = (0.15, 0.25, 0.30, 0.30),
) -> dict:
    """Map most sensitive layers (high S=λ/n) to bigger bit-widths."""
    assert len(proportions) == len(bit_bins) - 1
    L = len(layers_ordered)
    cuts = [int(L * p) for p in proportions]
    idxs = [0]
    for c in cuts:
        idxs.append(idxs[-1] + c)
    idxs.append(L)

    assign = {}
    for i in range(len(bit_bins)):
        start, end = idxs[i], idxs[i + 1]
        for name in layers_ordered[start:end]:
            assign[name] = bit_bins[i]
    return assign

def hawq_quantize_network(
    model: nn.Module,
    masks: Dict,
    train_loader: DataLoader,
    device: torch.device,
    bit_bins: tuple = (8, 6, 4, 3, 2),
    proportions: tuple = (0.15, 0.25, 0.30, 0.30),
    power_iters: int = 8,
    batches_per_layer: int = 2,
    loss_fn = F.cross_entropy,
    verbose: bool = True,
) -> Tuple[Dict, Dict]:
    """
    HAWQ-style mixed-precision quantization.
    Estimates Hessian eigenvalues to assign bits intelligently.
    Only considers non-pruned weights (respects masks).
    """
    model.eval()
    layers = _layer_params(model, masks)
    if verbose:
        print(f"\n{'='*60}")
        print("HAWQ: Estimating Layer Sensitivities (Non-Pruned Weights Only)")
        print(f"{'='*60}")

    data_iter = iter(train_loader)

    lambdas = {}
    sensitivities = {}
    for name, m in layers:
        mask = masks[name]
        lam = _top_hessian_eig_for_layer(
            model, m, mask, loss_fn, data_iter, device,
            power_iters=power_iters, batches=batches_per_layer
        )
        lambdas[name] = lam
        # Count only non-pruned parameters
        n_i = float(mask.sum().item())
        S_i = lam / (n_i + 1e-12)
        sensitivities[name] = S_i
        if verbose:
            total_params = m.weight.numel()
            sparsity = (1 - n_i / total_params) * 100
            print(f"{name:<40} λ≈{lam:9.4e}  n={int(n_i):>8} ({sparsity:.1f}% pruned)  S=λ/n≈{S_i:9.4e}")

    ordered = sorted(sensitivities.keys(), key=lambda k: sensitivities[k], reverse=True)

    bit_assign = _assign_bits_by_sensitivity(sensitivities, ordered, bit_bins, proportions)
    if verbose:
        print(f"\n{'='*60}")
        print("HAWQ Bit Assignment")
        print(f"{'='*60}")
        for name in ordered:
            print(f"{name:<40} -> {bit_assign[name]} bits")

    if verbose:
        print(f"\n{'='*60}")
        print("Quantizing with HAWQ Assignment")
        print(f"{'='*60}")

    codebooks = {}
    for name, m in layers:
        n_bits = bit_assign[name]
        qW, C = quantize_layer(m.weight.data, masks[name], n_bits)
        m.weight.data = qW
        codebooks[name] = C
        if verbose:
            print(f"{name:<40} quantized to {n_bits} bits ({2**n_bits} clusters)")

    return codebooks, bit_assign


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def deep_compress_mobilenetv2(checkpoint_path: str, 
                              output_dir: str = './compressed_model',
                              prune_ratio: float = 0.5,
                              conv_bits: int = 8,
                              fc_bits: int = 5):
    """
    Complete Deep Compression pipeline for MobileNetV2.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = build_cifar10_loaders(
        data_dir="./data", batch_size=128, workers=4
    )
    
    # Initialize model
    cfg = MV2Config(num_classes=10, width_mult=1.0)
    model = MobileNetV2(cfg).to(device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"\nLoaded checkpoint: {checkpoint_path}")
    
    # Baseline accuracy
    baseline_loss, baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    
    # Stage 1: Pruning
    masks = prune_network(model, sparsity=prune_ratio, exclude_first_last=True)
    pruned_loss, pruned_acc = evaluate(model, test_loader, device)
    print(f"After Pruning: {pruned_acc:.2f}% "
          f"(drop: {baseline_acc - pruned_acc:.2f}%)")
    
    # Stage 2: Quantization
    if quant_mode := getattr(args, 'quant_mode', 'uniform') == 'uniform':
        codebooks = quantize_network(model, masks, conv_bits, fc_bits)
        bit_assign = None
        quant_tag = f"uniform_c{conv_bits}b_fc{fc_bits}b"
    else:
        # HAWQ mixed-precision
        codebooks, bit_assign = hawq_quantize_network(
            model, masks, train_loader, device,
            bit_bins=args.bit_bins,
            proportions=args.bin_proportions,
            power_iters=args.power_iters,
            batches_per_layer=args.eig_batches,
            verbose=True,
        )
        # Optional: compute a weighted average bits (by nnz) for logging/filename
        total_nnz = sum(int(masks[n].sum().item()) for n in masks)
        avg_bits = 0.0
        if total_nnz > 0:
            for n, m in model.named_modules():
                if n in masks and isinstance(m, (nn.Conv2d, nn.Linear)):
                    avg_bits += bit_assign[n] * int(masks[n].sum().item())
            avg_bits /= total_nnz
        quant_tag = f"hawq_avg{avg_bits:.1f}b"
    
    quant_loss, quant_acc = evaluate(model, test_loader, device)
    print(f"After Quantization: {quant_acc:.2f}% "
          f"(drop: {baseline_acc - quant_acc:.2f}%)")
    
    # --- Optional: Post-Training Activation Quantization (PTQ) ---
    if getattr(args, 'act_bits', 0) and args.act_bits > 0:
        print(f"\n{'='*60}\nACTIVATION PTQ: inserting {args.act_bits}-bit quantizers\n{'='*60}")
        _attach_act_quantizers(model, n_bits=args.act_bits, symmetric=args.act_symmetric)

        # Calibrate on training batches
        print(f"Calibrating activation ranges with {args.act_calib_batches} batches...")
        calibrate_activations(model, train_loader, device, num_batches=args.act_calib_batches)

        # Evaluate with fake-quant enabled
        ptq_loss, ptq_acc = evaluate(model, test_loader, device)
        print(f"After Activation PTQ: {ptq_acc:.2f}% "
              f"(drop vs baseline: {baseline_acc - ptq_acc:.2f}%)")
        final_acc = ptq_acc
    else:
        final_acc = quant_acc
        print(f"Final accuracy (without fine-tuning): {final_acc:.2f}%")
    
    # Calculate weight bits
    total_weight_bits = sum(
        int(masks[n].sum().item()) * 
        (0 if codebooks[n].numel() <= 1 else math.ceil(math.log2(int(codebooks[n].numel()))))
        for n, m in model.named_modules()
        if n in masks and n in codebooks and isinstance(m, (nn.Conv2d, nn.Linear))
    )
    
    # Calculate activation bits (must be after forward passes!)
    if getattr(args, 'act_bits', 0) and args.act_bits > 0:
        total_activation_bits = total_activation_bits_from_quantizers(model)
    else:
        # No activation quantization - use FP32
        total_activation_bits = orig_activation_bits(model, input_shape=(1,3,32,32), act_bits=32)
    
    # Calculate original sizes
    orig_weight_bits = sum(p.numel() * 32 for p in model.parameters())  # FP32
    original_activation_bits = orig_activation_bits(model, input_shape=(1,3,32,32), act_bits=32)
    
    print(f"\n{'='*60}")
    print("MEMORY COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Weight bits:      {total_weight_bits/8/1024/1024:.2f} MB (compressed)")
    print(f"Activation bits:  {total_activation_bits/8/1024/1024:.2f} MB (compressed)")
    print(f"Original weights: {orig_weight_bits/8/1024/1024:.2f} MB (FP32)")
    print(f"Original activations: {original_activation_bits/8/1024/1024:.2f} MB (FP32)")
    print(f"\nWeight Compression:     {orig_weight_bits/total_weight_bits:.2f}×")
    print(f"Activation Compression: {original_activation_bits/total_activation_bits:.2f}×")
    print(f"Total Compression:      {(original_activation_bits+orig_weight_bits)/(total_activation_bits+total_weight_bits):.2f}×")
    print(f"Compressed Model size:{((total_activation_bits+total_weight_bits)/8/1024/1024):.2f}")
    
    return final_acc

def total_activation_bits_from_quantizers(model: nn.Module, batch_size: int = 1) -> int:
    """
    Sum activation storage bits based on attached quantizers.
    Divides by actual batch size to get per-sample memory.
    """
    total_bits = 0
    n_quantizers = 0
    
    for m in model.modules():
        if isinstance(m, ActivationPTQ):
            n_quantizers += 1
            if hasattr(m, 'last_numel') and m.last_numel > 0:
                # Divide by batch size to get per-sample memory
                per_sample_numel = m.last_numel // m.last_shape[0] if len(m.last_shape) > 0 else m.last_numel
                total_bits += per_sample_numel * m.n_bits * batch_size
            else:
                print(f"WARNING: Quantizer found but no shape tracked!")
                return 0
    
    if n_quantizers == 0:
        print("WARNING: No activation quantizers found!")
        return 0
    
    return total_bits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Compression for MobileNetV2')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./compressed_models',
                       help='Directory to save compressed models')
    parser.add_argument('--sparsity', type=float, default=0.5,
                       help='Pruning sparsity ratio (0.5 = 50%% pruned)')
    parser.add_argument('--conv-bits', type=int, default=8,
                       help='Number of bits for CONV layer quantization')
    parser.add_argument('--fc-bits', type=int, default=5,
                       help='Number of bits for FC layer quantization')
    parser.add_argument('--quant-mode', type=str, default='uniform',
                        choices=['uniform', 'hawq'],
                        help='Quantization mode: uniform (conv_bits/fc_bits) or hawq (mixed precision)')
    parser.add_argument('--bit-bins', type=str, default='8,6,4,3,2',
                        help='HAWQ: comma-separated bit bins (high→low), e.g. "8,6,4,3,2"')
    parser.add_argument('--bin-proportions', type=str, default='0.15,0.25,0.30,0.30',
                        help='HAWQ: comma-separated proportions for all but the last bin, e.g. "0.15,0.25,0.30,0.30"')
    parser.add_argument('--power-iters', type=int, default=8,
                        help='HAWQ: power iterations per layer for top Hessian eigenvalue')
    parser.add_argument('--eig-batches', type=int, default=2,
                        help='HAWQ: small number of batches per layer for eigen estimation')
    parser.add_argument('--act-bits', type=int, default=0,
                        help='Post-training activation bits (0=disabled)')
    parser.add_argument('--act-symmetric', action='store_true',
                        help='Use symmetric activation quantization (default: asym)')
    parser.add_argument('--act-calib-batches', type=int, default=20,
                        help='Batches to use for activation-range calibration')
    args = parser.parse_args()
    def _parse_int_list(s): 
        return tuple(int(x.strip()) for x in s.split(',') if x.strip())

    def _parse_float_list(s):
        return tuple(float(x.strip()) for x in s.split(',') if x.strip())

    args.bit_bins = _parse_int_list(args.bit_bins)            # e.g., (8,6,4,3,2)
    args.bin_proportions = _parse_float_list(args.bin_proportions)  # e.g., (0.15,0.25,0.30,0.30)

    print(f"\n{'='*60}")
    print("DEEP COMPRESSION CONFIGURATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sparsity: {args.sparsity*100:.0f}%")
    print(f"Quant mode: {args.quant_mode}")
    if args.quant_mode == 'uniform':
        print(f"Uniform bits — CONV {args.conv_bits}, FC {args.fc_bits}")
    else:
        print(f"HAWQ bits: {args.bit_bins} with proportions {args.bin_proportions}")
        print(f"HAWQ power iters: {args.power_iters}, eig batches: {args.eig_batches}")

    print(f"{'='*60}\n")
    
    deep_compress_mobilenetv2(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        prune_ratio=args.sparsity,
        conv_bits=args.conv_bits,
        fc_bits=args.fc_bits
    )