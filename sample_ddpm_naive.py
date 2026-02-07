#!/usr/bin/env python3
"""
sample_ddpm_naive.py
====================
A pure-Python (no external libraries) implementation of DDPM unconditional
sampling that is *structurally identical* to the PyTorch version defined in
mindiffusion/ddpm.py and mindiffusion/unet.py.

It reads the same JSON weight file (converted from a PyTorch .pth) and
produces a sampled image saved as a PNG file.

Only Python standard-library modules are used: math, json, random, struct, zlib, os, argparse.
"""

import argparse
import json
import math
import os
import random
import struct
import zlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ============================================================================
# 1. Minimal Tensor class — flat storage + shape, supporting the ops we need
# ============================================================================

class Tensor:
    """
    A minimal multi-dimensional tensor backed by a flat Python list of floats.
    Supports the subset of operations required to run the NaiveUnet DDPM sampler.
    """

    __slots__ = ("data", "shape", "strides", "offset", "_size")

    # ---- construction ------------------------------------------------------

    def __init__(
        self,
        data: List[float],
        shape: Tuple[int, ...],
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
    ) -> None:
        self.data = data
        self.shape = tuple(shape)
        self.offset = offset
        if strides is not None:
            self.strides = tuple(strides)
        else:
            self.strides = self._compute_strides(self.shape)
        self._size = 1
        for s in self.shape:
            self._size *= s

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    # ---- factories ---------------------------------------------------------

    @staticmethod
    def zeros(shape: Tuple[int, ...]) -> "Tensor":
        size = 1
        for s in shape:
            size *= s
        return Tensor([0.0] * size, shape)

    @staticmethod
    def ones(shape: Tuple[int, ...]) -> "Tensor":
        size = 1
        for s in shape:
            size *= s
        return Tensor([1.0] * size, shape)

    @staticmethod
    def full(shape: Tuple[int, ...], value: float) -> "Tensor":
        size = 1
        for s in shape:
            size *= s
        return Tensor([value] * size, shape)

    @staticmethod
    def randn(shape: Tuple[int, ...]) -> "Tensor":
        """Standard normal via Box-Muller."""
        size = 1
        for s in shape:
            size *= s
        data = [0.0] * size
        i = 0
        while i < size:
            u1 = random.random()
            while u1 == 0.0:
                u1 = random.random()
            u2 = random.random()
            mag = math.sqrt(-2.0 * math.log(u1))
            z0 = mag * math.cos(2.0 * math.pi * u2)
            z1 = mag * math.sin(2.0 * math.pi * u2)
            data[i] = z0
            i += 1
            if i < size:
                data[i] = z1
                i += 1
        return Tensor(data, shape)

    # ---- helpers -----------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return self._size

    def clone(self) -> "Tensor":
        return Tensor(list(self._contiguous_data()), self.shape)

    def _contiguous_data(self) -> List[float]:
        """Return flat data in row-major order."""
        if self.strides == self._compute_strides(self.shape) and self.offset == 0:
            return self.data[: self._size]
        out: List[float] = [0.0] * self._size
        for idx in range(self._size):
            out[idx] = self._flat_get(idx)
        return out

    def contiguous(self) -> "Tensor":
        if self.strides == self._compute_strides(self.shape) and self.offset == 0:
            return self
        return Tensor(self._contiguous_data(), self.shape)

    def _flat_get(self, flat_idx: int) -> float:
        """Get element by row-major flat index, respecting strides/offset."""
        real = self.offset
        rem = flat_idx
        for i in range(len(self.shape)):
            sz = self.shape[i]
            coord = rem // (self._size // (1 if i == 0 else 1))  # wrong approach
            # simpler: unflatten
            pass
        # Rewrite properly:
        real = self.offset
        rem = flat_idx
        for i in range(len(self.shape)):
            dim_size = 1
            for j in range(i + 1, len(self.shape)):
                dim_size *= self.shape[j]
            coord = rem // dim_size
            rem = rem % dim_size
            real += coord * self.strides[i]
        return self.data[real]

    def _flat_set(self, flat_idx: int, value: float) -> None:
        real = self.offset
        rem = flat_idx
        for i in range(len(self.shape)):
            dim_size = 1
            for j in range(i + 1, len(self.shape)):
                dim_size *= self.shape[j]
            coord = rem // dim_size
            rem = rem % dim_size
            real += coord * self.strides[i]
        self.data[real] = value

    # ---- indexing (simple integer or slice) ----------------------------------

    def __getitem__(self, key):
        if isinstance(key, int):
            if self.ndim == 1:
                return self.data[self.offset + key * self.strides[0]]
            new_shape = self.shape[1:]
            new_strides = self.strides[1:]
            new_offset = self.offset + key * self.strides[0]
            return Tensor(self.data, new_shape, new_strides, new_offset)
        raise NotImplementedError(f"Indexing with {type(key)} not supported")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if self.ndim == 1:
                self.data[self.offset + key * self.strides[0]] = float(value)
                return
            sub = self[key]
            if isinstance(value, (int, float)):
                for i in range(sub.numel()):
                    sub._flat_set(i, float(value))
            elif isinstance(value, Tensor):
                src = value.contiguous()
                for i in range(sub.numel()):
                    sub._flat_set(i, src.data[i])
            return
        raise NotImplementedError

    # ---- reshape / view ----------------------------------------------------

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        data = self._contiguous_data()
        return Tensor(data, new_shape)

    def view(self, *shape: int) -> "Tensor":
        # Handle -1
        total = self._size
        neg_idx = -1
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx >= 0:
            shape_list = list(shape)
            shape_list[neg_idx] = total // known
            shape = tuple(shape_list)
        return self.reshape(shape)

    # ---- arithmetic (element-wise) -----------------------------------------

    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            out_data = self._contiguous_data()
            return Tensor([x + other for x in out_data], self.shape)
        other_tensor: Tensor = other
        return _broadcast_binary(self, other_tensor, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            out_data = self._contiguous_data()
            return Tensor([x - other for x in out_data], self.shape)
        return _broadcast_binary(self, other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            out_data = self._contiguous_data()
            return Tensor([other - x for x in out_data], self.shape)
        return other.__sub__(self)

    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            out_data = self._contiguous_data()
            return Tensor([x * other for x in out_data], self.shape)
        return _broadcast_binary(self, other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            out_data = self._contiguous_data()
            return Tensor([x / other for x in out_data], self.shape)
        return _broadcast_binary(self, other, lambda a, b: a / b)

    def __neg__(self) -> "Tensor":
        out_data = self._contiguous_data()
        return Tensor([-x for x in out_data], self.shape)

    # ---- math functions ---------------------------------------------------

    def sin(self) -> "Tensor":
        d = self._contiguous_data()
        return Tensor([math.sin(x) for x in d], self.shape)

    def sqrt(self) -> "Tensor":
        d = self._contiguous_data()
        return Tensor([math.sqrt(x) for x in d], self.shape)

    def log(self) -> "Tensor":
        d = self._contiguous_data()
        return Tensor([math.log(x) for x in d], self.shape)

    def exp(self) -> "Tensor":
        d = self._contiguous_data()
        return Tensor([math.exp(x) for x in d], self.shape)

    def relu(self) -> "Tensor":
        d = self._contiguous_data()
        return Tensor([max(0.0, x) for x in d], self.shape)

    def cumsum(self, dim: int = 0) -> "Tensor":
        t = self.contiguous()
        out_data = list(t.data[:t._size])
        if t.ndim == 1:
            for i in range(1, t.shape[0]):
                out_data[i] = out_data[i - 1] + out_data[i]
        else:
            raise NotImplementedError("cumsum only for 1D")
        return Tensor(out_data, t.shape)

    # ---- reductions -------------------------------------------------------

    def sum(self) -> float:
        return sum(self._contiguous_data())

    def mean(self) -> float:
        d = self._contiguous_data()
        return sum(d) / len(d)

    def var(self) -> float:
        d = self._contiguous_data()
        m = sum(d) / len(d)
        return sum((x - m) ** 2 for x in d) / len(d)

    def mean_over(self, indices: List[int]) -> "Tensor":
        """Mean over the given flat indices (used for group-norm channels)."""
        # Not used directly; group norm is implemented differently.
        raise NotImplementedError

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"


# ---- broadcast helper -----------------------------------------------------

def _broadcast_shapes(
    s1: Tuple[int, ...], s2: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute broadcast shape (NumPy-style)."""
    nd = max(len(s1), len(s2))
    p1 = (1,) * (nd - len(s1)) + s1
    p2 = (1,) * (nd - len(s2)) + s2
    out = []
    for a, b in zip(p1, p2):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise ValueError(f"Cannot broadcast {s1} and {s2}")
    return tuple(out)


def _broadcast_binary(a: Tensor, b: Tensor, op) -> Tensor:
    out_shape = _broadcast_shapes(a.shape, b.shape)
    nd = len(out_shape)
    pa = (1,) * (nd - a.ndim) + a.shape
    pb = (1,) * (nd - b.ndim) + b.shape

    a_c = a.contiguous()
    b_c = b.contiguous()

    out_size = 1
    for s in out_shape:
        out_size *= s
    out_data = [0.0] * out_size

    # Precompute strides for output, a, b
    def strides_for(sh):
        st = [1] * len(sh)
        for i in range(len(sh) - 2, -1, -1):
            st[i] = st[i + 1] * sh[i + 1]
        return st

    out_st = strides_for(out_shape)
    pa_st = strides_for(pa)
    pb_st = strides_for(pb)

    a_strides_orig = Tensor._compute_strides(a.shape)
    b_strides_orig = Tensor._compute_strides(b.shape)

    for flat_idx in range(out_size):
        # Unflatten flat_idx in out_shape
        rem = flat_idx
        a_idx = 0
        b_idx = 0
        for d in range(nd):
            coord = rem // out_st[d] if d < nd - 1 else rem
            rem = rem % out_st[d] if d < nd - 1 else 0
            a_coord = coord if pa[d] != 1 else 0
            b_coord = coord if pb[d] != 1 else 0
            # map to original strides
            a_d_orig = d - (nd - a.ndim)
            b_d_orig = d - (nd - b.ndim)
            if a_d_orig >= 0:
                a_idx += a_coord * a_strides_orig[a_d_orig]
            if b_d_orig >= 0:
                b_idx += b_coord * b_strides_orig[b_d_orig]
        out_data[flat_idx] = op(a_c.data[a_idx], b_c.data[b_idx])

    return Tensor(out_data, out_shape)


# ============================================================================
# 2. Neural-network layer functions (pure Python)
# ============================================================================

def conv2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """
    2-D convolution. inp: (N,C_in,H,W), weight: (C_out,C_in,kH,kW).
    """
    inp = inp.contiguous()
    weight = weight.contiguous()

    N, C_in, H, W = inp.shape
    C_out, C_in_w, kH, kW = weight.shape
    assert C_in == C_in_w

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    out_data = [0.0] * (N * C_out * H_out * W_out)

    inp_d = inp.data
    w_d = weight.data
    bias_d = bias.contiguous().data if bias is not None else None

    # Precompute strides
    inp_s0 = C_in * H * W
    inp_s1 = H * W
    inp_s2 = W

    w_s0 = C_in_w * kH * kW
    w_s1 = kH * kW
    w_s2 = kW

    out_s0 = C_out * H_out * W_out
    out_s1 = H_out * W_out
    out_s2 = W_out

    for n in range(N):
        for co in range(C_out):
            bias_val = bias_d[co] if bias_d is not None else 0.0
            for oh in range(H_out):
                for ow in range(W_out):
                    val = bias_val
                    ih_start = oh * stride - padding
                    iw_start = ow * stride - padding
                    for ci in range(C_in):
                        for kh in range(kH):
                            ih = ih_start + kh
                            if ih < 0 or ih >= H:
                                continue
                            for kw in range(kW):
                                iw = iw_start + kw
                                if iw < 0 or iw >= W:
                                    continue
                                val += (
                                    inp_d[n * inp_s0 + ci * inp_s1 + ih * inp_s2 + iw]
                                    * w_d[co * w_s0 + ci * w_s1 + kh * w_s2 + kw]
                                )
                    out_data[n * out_s0 + co * out_s1 + oh * out_s2 + ow] = val

    return Tensor(out_data, (N, C_out, H_out, W_out))


def conv_transpose2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """
    2-D transposed convolution. inp: (N,C_in,H,W), weight: (C_in,C_out,kH,kW).
    Output shape: (N, C_out, (H-1)*stride - 2*padding + kH, (W-1)*stride - 2*padding + kW)
    """
    inp = inp.contiguous()
    weight = weight.contiguous()

    N, C_in, H, W = inp.shape
    C_in_w, C_out, kH, kW = weight.shape
    assert C_in == C_in_w

    H_out = (H - 1) * stride - 2 * padding + kH
    W_out = (W - 1) * stride - 2 * padding + kW

    out_size = N * C_out * H_out * W_out
    out_data = [0.0] * out_size

    inp_d = inp.data
    w_d = weight.data
    bias_d = bias.contiguous().data if bias is not None else None

    inp_s0 = C_in * H * W
    inp_s1 = H * W
    inp_s2 = W

    w_s0 = C_out * kH * kW
    w_s1 = kH * kW
    w_s2 = kW

    out_s0 = C_out * H_out * W_out
    out_s1 = H_out * W_out
    out_s2 = W_out

    # Scatter: for each input pixel, scatter to output
    for n in range(N):
        for ci in range(C_in):
            for ih in range(H):
                for iw in range(W):
                    inp_val = inp_d[n * inp_s0 + ci * inp_s1 + ih * inp_s2 + iw]
                    if inp_val == 0.0:
                        continue
                    for co in range(C_out):
                        for kh in range(kH):
                            oh = ih * stride - padding + kh
                            if oh < 0 or oh >= H_out:
                                continue
                            for kw in range(kW):
                                ow = iw * stride - padding + kw
                                if ow < 0 or ow >= W_out:
                                    continue
                                out_data[
                                    n * out_s0 + co * out_s1 + oh * out_s2 + ow
                                ] += (
                                    inp_val
                                    * w_d[ci * w_s0 + co * w_s1 + kh * w_s2 + kw]
                                )

    if bias_d is not None:
        for n in range(N):
            for co in range(C_out):
                b = bias_d[co]
                base = n * out_s0 + co * out_s1
                for j in range(H_out * W_out):
                    out_data[base + j] += b

    return Tensor(out_data, (N, C_out, H_out, W_out))


def group_norm(
    inp: Tensor, num_groups: int, weight: Tensor, bias: Tensor, eps: float = 1e-5
) -> Tensor:
    """
    Group Normalization. inp: (N, C, H, W).
    """
    inp = inp.contiguous()
    weight = weight.contiguous()
    bias_t = bias.contiguous()

    N, C, H, W = inp.shape
    assert C % num_groups == 0
    cpg = C // num_groups  # channels per group
    spatial = H * W

    out_data = list(inp.data[: N * C * H * W])

    for n in range(N):
        for g in range(num_groups):
            # Compute mean and var for this group
            c_start = g * cpg
            c_end = c_start + cpg
            total = 0.0
            count = cpg * spatial
            for c in range(c_start, c_end):
                base = n * C * spatial + c * spatial
                for j in range(spatial):
                    total += out_data[base + j]
            mean = total / count

            var_sum = 0.0
            for c in range(c_start, c_end):
                base = n * C * spatial + c * spatial
                for j in range(spatial):
                    diff = out_data[base + j] - mean
                    var_sum += diff * diff
            var = var_sum / count

            inv_std = 1.0 / math.sqrt(var + eps)

            for c in range(c_start, c_end):
                w = weight.data[c]
                b = bias_t.data[c]
                base = n * C * spatial + c * spatial
                for j in range(spatial):
                    out_data[base + j] = (out_data[base + j] - mean) * inv_std * w + b

    return Tensor(out_data, inp.shape)


def linear(inp: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    """
    Linear layer. inp: (..., in_features), weight: (out_features, in_features).
    """
    inp = inp.contiguous()
    weight = weight.contiguous()

    in_f = weight.shape[1]
    out_f = weight.shape[0]

    # Flatten all but last dim
    total_batch = inp.numel() // in_f
    out_data = [0.0] * (total_batch * out_f)

    w_d = weight.data
    inp_d = inp.data
    bias_d = bias.contiguous().data if bias is not None else None

    for b in range(total_batch):
        inp_off = b * in_f
        out_off = b * out_f
        for o in range(out_f):
            val = bias_d[o] if bias_d is not None else 0.0
            w_off = o * in_f
            for i in range(in_f):
                val += inp_d[inp_off + i] * w_d[w_off + i]
            out_data[out_off + o] = val

    out_shape = list(inp.shape[:-1]) + [out_f]
    return Tensor(out_data, tuple(out_shape))


def max_pool2d(inp: Tensor, kernel_size: int) -> Tensor:
    """2D max pooling."""
    inp = inp.contiguous()
    N, C, H, W = inp.shape
    kH = kW = kernel_size
    H_out = H // kH
    W_out = W // kW

    out_data = [0.0] * (N * C * H_out * W_out)
    inp_d = inp.data

    s0 = C * H * W
    s1 = H * W
    s2 = W
    os0 = C * H_out * W_out
    os1 = H_out * W_out
    os2 = W_out

    for n in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    max_val = -1e30
                    for kh in range(kH):
                        for kw in range(kW):
                            ih = oh * kH + kh
                            iw = ow * kW + kw
                            v = inp_d[n * s0 + c * s1 + ih * s2 + iw]
                            if v > max_val:
                                max_val = v
                    out_data[n * os0 + c * os1 + oh * os2 + ow] = max_val

    return Tensor(out_data, (N, C, H_out, W_out))


def avg_pool2d(inp: Tensor, kernel_size: int) -> Tensor:
    """2D average pooling."""
    inp = inp.contiguous()
    N, C, H, W = inp.shape
    kH = kW = kernel_size
    H_out = H // kH
    W_out = W // kW
    count = kH * kW

    out_data = [0.0] * (N * C * H_out * W_out)
    inp_d = inp.data

    s0 = C * H * W
    s1 = H * W
    s2 = W
    os0 = C * H_out * W_out
    os1 = H_out * W_out
    os2 = W_out

    for n in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    total = 0.0
                    for kh in range(kH):
                        for kw in range(kW):
                            ih = oh * kH + kh
                            iw = ow * kW + kw
                            total += inp_d[n * s0 + c * s1 + ih * s2 + iw]
                    out_data[n * os0 + c * os1 + oh * os2 + ow] = total / count

    return Tensor(out_data, (N, C, H_out, W_out))


def relu(inp: Tensor) -> Tensor:
    d = inp.contiguous().data
    return Tensor([max(0.0, x) for x in d[: inp.numel()]], inp.shape)


def cat_channels(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate along dim=1 (channel dim). Both must be (N, C_a, H, W) and (N, C_b, H, W)."""
    a = a.contiguous()
    b = b.contiguous()
    N, Ca, H, W = a.shape
    _, Cb, _, _ = b.shape
    C_out = Ca + Cb
    spatial = H * W

    out_data = [0.0] * (N * C_out * spatial)

    a_d = a.data
    b_d = b.data

    for n in range(N):
        # copy a channels
        a_base = n * Ca * spatial
        out_base = n * C_out * spatial
        for i in range(Ca * spatial):
            out_data[out_base + i] = a_d[a_base + i]
        # copy b channels
        b_base = n * Cb * spatial
        out_base2 = out_base + Ca * spatial
        for i in range(Cb * spatial):
            out_data[out_base2 + i] = b_d[b_base + i]

    return Tensor(out_data, (N, C_out, H, W))


# ============================================================================
# 3. Network modules (mirroring PyTorch nn.Module structure)
# ============================================================================

class Module:
    """Base module — just holds named weights."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Conv3(Module):
    """Mirrors mindiffusion.unet.Conv3."""

    def __init__(self, prefix: str, sd: Dict[str, Tensor], is_res: bool = False):
        self.is_res = is_res
        # main: Conv2d -> GroupNorm -> (ReLU)
        self.main_conv_w = sd[f"{prefix}.main.0.weight"]
        self.main_conv_b = sd[f"{prefix}.main.0.bias"]
        self.main_gn_w = sd[f"{prefix}.main.1.weight"]
        self.main_gn_b = sd[f"{prefix}.main.1.bias"]

        # conv: Conv2d -> GN -> ReLU -> Conv2d -> GN -> ReLU
        self.conv0_w = sd[f"{prefix}.conv.0.weight"]
        self.conv0_b = sd[f"{prefix}.conv.0.bias"]
        self.conv0_gn_w = sd[f"{prefix}.conv.1.weight"]
        self.conv0_gn_b = sd[f"{prefix}.conv.1.bias"]

        self.conv1_w = sd[f"{prefix}.conv.3.weight"]
        self.conv1_b = sd[f"{prefix}.conv.3.bias"]
        self.conv1_gn_w = sd[f"{prefix}.conv.4.weight"]
        self.conv1_gn_b = sd[f"{prefix}.conv.4.bias"]

    def forward(self, x: Tensor) -> Tensor:
        # main
        x = conv2d(x, self.main_conv_w, self.main_conv_b, stride=1, padding=1)
        x = group_norm(x, 8, self.main_gn_w, self.main_gn_b)
        x = relu(x)

        if self.is_res:
            # conv branch
            h = conv2d(x, self.conv0_w, self.conv0_b, stride=1, padding=1)
            h = group_norm(h, 8, self.conv0_gn_w, self.conv0_gn_b)
            h = relu(h)
            h = conv2d(h, self.conv1_w, self.conv1_b, stride=1, padding=1)
            h = group_norm(h, 8, self.conv1_gn_w, self.conv1_gn_b)
            h = relu(h)
            x = (x + h) * (1.0 / 1.414)
        else:
            x = conv2d(x, self.conv0_w, self.conv0_b, stride=1, padding=1)
            x = group_norm(x, 8, self.conv0_gn_w, self.conv0_gn_b)
            x = relu(x)
            x = conv2d(x, self.conv1_w, self.conv1_b, stride=1, padding=1)
            x = group_norm(x, 8, self.conv1_gn_w, self.conv1_gn_b)
            x = relu(x)
        return x


class UnetDown(Module):
    """Mirrors mindiffusion.unet.UnetDown: Conv3 + MaxPool2d(2)."""

    def __init__(self, prefix: str, sd: Dict[str, Tensor]):
        self.conv3 = Conv3(f"{prefix}.model.0", sd, is_res=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv3(x)
        x = max_pool2d(x, 2)
        return x


class UnetUp(Module):
    """Mirrors mindiffusion.unet.UnetUp: ConvTranspose2d + Conv3 + Conv3."""

    def __init__(self, prefix: str, sd: Dict[str, Tensor]):
        # ConvTranspose2d(in, out, 2, 2)
        self.ct_w = sd[f"{prefix}.model.0.weight"]
        self.ct_b = sd[f"{prefix}.model.0.bias"]
        # Two Conv3 blocks (not residual)
        self.conv3_1 = Conv3(f"{prefix}.model.1", sd, is_res=False)
        self.conv3_2 = Conv3(f"{prefix}.model.2", sd, is_res=False)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = cat_channels(x, skip)
        x = conv_transpose2d(x, self.ct_w, self.ct_b, stride=2, padding=0)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        return x


class TimeSiren(Module):
    """Mirrors mindiffusion.unet.TimeSiren."""

    def __init__(self, prefix: str, sd: Dict[str, Tensor]):
        self.lin1_w = sd[f"{prefix}.lin1.weight"]  # (emb_dim, 1), no bias
        self.lin2_w = sd[f"{prefix}.lin2.weight"]  # (emb_dim, emb_dim)
        self.lin2_b = sd[f"{prefix}.lin2.bias"]  # (emb_dim,)

    def forward(self, x: Tensor) -> Tensor:
        # x comes in as (N, 1) — reshape if needed
        if x.ndim == 1:
            x = x.view(-1, 1)
        elif x.ndim == 0 or (x.ndim == 2 and x.shape[1] != 1):
            x = x.view(-1, 1)
        # lin1 (no bias)
        x = linear(x, self.lin1_w, None)
        x = x.sin()
        x = linear(x, self.lin2_w, self.lin2_b)
        return x


class NaiveUnet(Module):
    """Mirrors mindiffusion.unet.NaiveUnet."""

    def __init__(self, sd: Dict[str, Tensor], n_feat: int = 128):
        self.n_feat = n_feat
        prefix = "eps_model"

        self.init_conv = Conv3(f"{prefix}.init_conv", sd, is_res=True)

        self.down1 = UnetDown(f"{prefix}.down1", sd)
        self.down2 = UnetDown(f"{prefix}.down2", sd)
        self.down3 = UnetDown(f"{prefix}.down3", sd)

        # to_vec: AvgPool2d(4) + ReLU (no learned params)

        self.timeembed = TimeSiren(f"{prefix}.timeembed", sd)

        # up0: ConvTranspose2d(2*n_feat, 2*n_feat, 4, 4) + GN + ReLU
        self.up0_ct_w = sd[f"{prefix}.up0.0.weight"]
        self.up0_ct_b = sd[f"{prefix}.up0.0.bias"]
        self.up0_gn_w = sd[f"{prefix}.up0.1.weight"]
        self.up0_gn_b = sd[f"{prefix}.up0.1.bias"]

        self.up1 = UnetUp(f"{prefix}.up1", sd)
        self.up2 = UnetUp(f"{prefix}.up2", sd)
        self.up3 = UnetUp(f"{prefix}.up3", sd)

        # out: Conv2d
        self.out_w = sd[f"{prefix}.out.weight"]
        self.out_b = sd[f"{prefix}.out.bias"]

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x0 = self.init_conv(x)

        d1 = self.down1(x0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        # to_vec: AvgPool2d(4) + ReLU
        thro = avg_pool2d(d3, 4)
        thro = relu(thro)

        # Time embedding: (N, 2*n_feat) -> (N, 2*n_feat, 1, 1)
        temb = self.timeembed(t)
        N_batch = temb.shape[0]
        temb_4d = temb.view(N_batch, self.n_feat * 2, 1, 1)

        # thro + temb (broadcast over spatial)
        thro = thro + temb_4d

        # up0: ConvTranspose2d(2*n_feat, 2*n_feat, 4, 4) stride=4
        thro = conv_transpose2d(thro, self.up0_ct_w, self.up0_ct_b, stride=4, padding=0)
        thro = group_norm(thro, 8, self.up0_gn_w, self.up0_gn_b)
        thro = relu(thro)

        up1 = self.up1(thro, d3)
        up1 = up1 + temb_4d  # broadcast add

        up2 = self.up2(up1, d2)
        up3 = self.up3(up2, d1)

        out = cat_channels(up3, x0)
        out = conv2d(out, self.out_w, self.out_b, stride=1, padding=1)

        return out


# ============================================================================
# 4. DDPM schedules & sampling
# ============================================================================

def ddpm_schedules(
    beta1: float, beta2: float, T: int
) -> Dict[str, List[float]]:
    """
    Returns pre-computed DDPM schedules as plain Python lists (length T+1).
    """
    beta_t = [(beta2 - beta1) * i / T + beta1 for i in range(T + 1)]
    sqrt_beta_t = [math.sqrt(b) for b in beta_t]
    alpha_t = [1.0 - b for b in beta_t]
    log_alpha_t = [math.log(a) for a in alpha_t]

    # cumsum of log_alpha_t
    cum = [0.0] * (T + 1)
    cum[0] = log_alpha_t[0]
    for i in range(1, T + 1):
        cum[i] = cum[i - 1] + log_alpha_t[i]
    alphabar_t = [math.exp(c) for c in cum]

    sqrtab = [math.sqrt(ab) for ab in alphabar_t]
    oneover_sqrta = [1.0 / math.sqrt(a) for a in alpha_t]
    sqrtmab = [math.sqrt(1.0 - ab) for ab in alphabar_t]
    mab_over_sqrtmab = [(1.0 - a) / sm if sm > 0 else 0.0 for a, sm in zip(alpha_t, sqrtmab)]

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }


def ddpm_sample(
    unet: NaiveUnet,
    n_sample: int,
    size: Tuple[int, ...],
    schedules: Dict[str, List[float]],
    n_T: int,
) -> Tensor:
    """
    DDPM sampling (Algorithm 2 from Ho et al.).
    """
    # Start from pure noise
    x_i = Tensor.randn((n_sample,) + size)

    oneover_sqrta = schedules["oneover_sqrta"]
    mab_over_sqrtmab = schedules["mab_over_sqrtmab"]
    sqrt_beta_t = schedules["sqrt_beta_t"]

    last_cost = None
    for i in range(n_T, 0, -1):
        
        start = time.perf_counter()

        # ===== print 上一次耗时 =====
        if last_cost is None:
            cost_str = "N/A"
        else:
            cost_str = f"{last_cost:.4f}s"

        print(
            f"Sampling step {n_T - i + 1}/{n_T} (t={i}) | last step cost: {cost_str}",
            end="\r"
        )

        # Build time tensor: (n_sample, 1) with value i/n_T
        t_val = i / n_T
        t_data = []
        for _ in range(n_sample):
            t_data.append(t_val)
        t_tensor = Tensor(t_data, (n_sample, 1))

        # Predict noise
        eps = unet(x_i, t_tensor)

        # Get schedule values for this timestep
        oos = oneover_sqrta[i]
        mos = mab_over_sqrtmab[i]
        sb = sqrt_beta_t[i]

        # z ~ N(0,1) if i > 1, else 0
        if i > 1:
            z = Tensor.randn((n_sample,) + size)
        else:
            z = Tensor.zeros((n_sample,) + size)

        # x_i = oneover_sqrta[i] * (x_i - eps * mab_over_sqrtmab[i]) + sqrt_beta_t[i] * z
        # Do element-wise
        x_d = x_i.contiguous().data
        e_d = eps.contiguous().data
        z_d = z.contiguous().data
        total = x_i.numel()
        new_data = [0.0] * total
        for j in range(total):
            new_data[j] = oos * (x_d[j] - e_d[j] * mos) + sb * z_d[j]
        x_i = Tensor(new_data, x_i.shape)
        
        # ===== 本轮结束，记录耗时 =====
        last_cost = time.perf_counter() - start

    print('Sampling done')
    return x_i


# ============================================================================
# 5. PNG writer (pure Python, using zlib for DEFLATE)
# ============================================================================

def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", _crc32(chunk_type + data))
    return length + chunk_type + data + crc


def save_png(path: str, pixels: List[List[Tuple[int, int, int]]], width: int, height: int) -> None:
    """
    Write an RGB PNG file.
    pixels: [row][col] = (R, G, B) each in 0..255.
    """
    # Signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _make_chunk(b"IHDR", ihdr_data)

    # IDAT
    raw_rows = bytearray()
    for row in pixels:
        raw_rows.append(0)  # filter type = None
        for r, g, b in row:
            raw_rows.append(r & 0xFF)
            raw_rows.append(g & 0xFF)
            raw_rows.append(b & 0xFF)
    compressed = zlib.compress(bytes(raw_rows))
    idat = _make_chunk(b"IDAT", compressed)

    # IEND
    iend = _make_chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(sig + ihdr + idat + iend)


def tensor_to_image(t: Tensor) -> List[List[Tuple[int, int, int]]]:
    """
    Convert a (3, H, W) tensor with values in [-1, 1] to an RGB pixel grid.
    Applies normalize: pixel = clamp((x + 1) / 2 * 255, 0, 255).
    """
    t = t.contiguous()
    assert t.ndim == 3 and t.shape[0] == 3
    C, H, W = t.shape
    d = t.data

    pixels = []
    for h in range(H):
        row = []
        for w in range(W):
            rgb = []
            for c in range(3):
                val = d[c * H * W + h * W + w]
                # normalize from [-1, 1] to [0, 255]
                val = (val + 1.0) / 2.0 * 255.0
                val = max(0.0, min(255.0, val))
                rgb.append(int(val + 0.5))
            row.append(tuple(rgb))
        pixels.append(row)
    return pixels


# ============================================================================
# 6. Weight loading from JSON
# ============================================================================

def load_weights_from_json(path: str) -> Dict[str, Tensor]:
    """Load JSON weights (converted from PyTorch state_dict) into our Tensor format."""
    print(f"Loading weights from {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def flatten(obj) -> List[float]:
        """Recursively flatten nested lists into a flat list of floats."""
        if isinstance(obj, (int, float)):
            return [float(obj)]
        result: List[float] = []
        for item in obj:
            result.extend(flatten(item))
        return result

    sd: Dict[str, Tensor] = {}
    total_keys = len(raw)
    for idx, (key, payload) in enumerate(raw.items()):
        shape = tuple(payload["shape"])
        data = flatten(payload["data"])          # <--- 关键修改
        sd[key] = Tensor(data, shape)
        if (idx + 1) % 20 == 0 or idx + 1 == total_keys:
            print(f"  Loaded {idx + 1}/{total_keys} parameters", end="\r")
    print()
    return sd


# ============================================================================
# 7. Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure-Python DDPM unconditional sampler (no external libs)."
    )
    parser.add_argument(
        "--json",
        default="ddpm_cifar.json",
        help="Path to JSON weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--out",
        default="contents/ddpm_sample_from_naiveddpm.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Load weights
    sd = load_weights_from_json(args.json)

    # Build model
    print("Building NaiveUnet ...")
    unet = NaiveUnet(sd, n_feat=128)

    # Compute schedules
    print("Computing DDPM schedules ...")
    schedules = ddpm_schedules(1e-4, 0.02, 1000)

    # Sample
    print("Starting DDPM sampling (1000 steps) — this will be SLOW in pure Python ...")
    samples = ddpm_sample(unet, n_sample=1, size=(3, 32, 32), schedules=schedules, n_T=1000)

    # Save image
    # samples shape: (1, 3, 32, 32) — take [0]
    sample = samples[0]  # (3, 32, 32)
    pixels = tensor_to_image(sample)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_png(args.out, pixels, 32, 32)
    print(f"Saved sample to {args.out}")


if __name__ == "__main__":
    main()