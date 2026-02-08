#!/usr/bin/env python3
"""
sample_ddpm_pure.py

Pure-Python (no PyTorch, no NumPy) implementation of DDPM sampling,
loading weights from a PyTorch-trained checkpoint.

Usage:
    python sample_ddpm_pure.py --checkpoint ./ddpm_mnist.pth --seed 123 --n-sample 1
"""

import argparse
import math
import os
import pickle
import random
import struct
import zipfile
import zlib
import io
from collections import OrderedDict


# ============================================================
# Part 0: Minimal N-dimensional array class
# ============================================================

class Tensor:
    """A minimal N-dimensional array backed by a flat Python list of floats."""

    def __init__(self, data=None, shape=None):
        if data is not None and shape is not None:
            self.data = data if isinstance(data, list) else list(data)
            self.shape = tuple(shape)
        elif data is not None:
            self.shape, self.data = self._flatten(data)
        else:
            self.data = []
            self.shape = (0,)
        assert len(self.data) == _prod(self.shape), (
            f"Data length {len(self.data)} != product of shape {self.shape} = {_prod(self.shape)}"
        )

    def _flatten(self, nested):
        if not isinstance(nested, (list, tuple)):
            return ((), [nested])
        if len(nested) == 0:
            return ((0,), [])
        sub_shapes = []
        flat = []
        for item in nested:
            s, f = Tensor(item)._get_shape_data()
            sub_shapes.append(s)
            flat.extend(f)
        for s in sub_shapes:
            assert s == sub_shapes[0]
        return ((len(nested),) + sub_shapes[0], flat)

    def _get_shape_data(self):
        return self.shape, self.data

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return len(self.data)

    def clone(self):
        return Tensor(list(self.data), self.shape)

    def reshape(self, *new_shape):
        return Tensor(self.data, new_shape)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.ndim == 1:
                return self.data[idx]
            inner = _prod(self.shape[1:])
            start = idx * inner
            return Tensor(self.data[start:start + inner], self.shape[1:])
        raise NotImplementedError

    def __repr__(self):
        if self.size <= 20:
            return f"Tensor(shape={self.shape}, data={self.data[:20]})"
        return f"Tensor(shape={self.shape}, data=[{self.data[0]:.4f}, ..., {self.data[-1]:.4f}])"


def _prod(seq):
    r = 1
    for s in seq:
        r *= s
    return r


def tensor_zeros(shape):
    return Tensor([0.0] * _prod(shape), shape)


# ============================================================
# Part 1: Random number generation (Box-Muller)
# ============================================================

class PureRandom:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def randn(self, *shape):
        n = _prod(shape)
        samples = []
        for _ in range((n + 1) // 2):
            u1 = self.rng.random()
            u2 = self.rng.random()
            while u1 == 0.0:
                u1 = self.rng.random()
            mag = math.sqrt(-2.0 * math.log(u1))
            z0 = mag * math.cos(2.0 * math.pi * u2)
            z1 = mag * math.sin(2.0 * math.pi * u2)
            samples.append(z0)
            samples.append(z1)
        return Tensor(samples[:n], shape)


# ============================================================
# Part 2: Load PyTorch checkpoint (state_dict)
# ============================================================

# ---- dtype specs ----
_DTYPE_INFO = {
    # name -> (struct_char, itemsize)
    "torch.float32":  ("f", 4),
    "torch.float64":  ("d", 8),
    "torch.float16":  ("e", 2),
    "torch.bfloat16": ("e", 2),  # approximate: read as float16
    "torch.int64":    ("q", 8),
    "torch.int32":    ("i", 4),
    "torch.int16":    ("h", 2),
    "torch.int8":     ("b", 1),
    "torch.uint8":    ("B", 1),
    # Legacy storage class names
    "FloatStorage":   ("f", 4),
    "DoubleStorage":  ("d", 8),
    "HalfStorage":    ("e", 2),
    "BFloat16Storage":("e", 2),
    "LongStorage":    ("q", 8),
    "IntStorage":     ("i", 4),
    "ShortStorage":   ("h", 2),
    "ByteStorage":    ("B", 1),
}


class _StorageTypePlaceholder:
    """Placeholder for torch.*Storage classes during unpickling."""
    def __init__(self, name):
        self.__name__ = name
        self.name = name
    def __repr__(self):
        return f"<StorageType:{self.name}>"
    def __call__(self, *args, **kwargs):
        return self


class _UntypedStoragePlaceholder:
    """Placeholder for torch.UntypedStorage."""
    def __init__(self, *args, **kwargs):
        pass
    def __repr__(self):
        return "<UntypedStorage>"


def _rebuild_tensor_v2(storage_obj, storage_offset, size, stride, requires_grad, backward_hooks):
    """Reconstruct a tensor from its storage (v2 protocol)."""
    flat_data = storage_obj["data"]
    shape = tuple(size)
    n = _prod(shape)
    if n == 0:
        return Tensor([], shape)

    stride = tuple(stride)

    # Check if contiguous
    expected_stride = []
    s = 1
    for dim in reversed(shape):
        expected_stride.append(s)
        s *= dim
    expected_stride = tuple(reversed(expected_stride))

    if stride == expected_stride:
        data = flat_data[storage_offset:storage_offset + n]
    else:
        data = [0.0] * n
        def _idx_to_storage(flat_idx):
            offset = storage_offset
            remainder = flat_idx
            for d in range(len(shape)):
                dim_size = shape[d]
                if d < len(shape) - 1:
                    rest = _prod(shape[d+1:])
                    coord = remainder // rest
                    remainder = remainder % rest
                else:
                    coord = remainder
                offset += coord * stride[d]
            return offset
        for i in range(n):
            data[i] = flat_data[_idx_to_storage(i)]

    return Tensor(list(data), shape)


def _rebuild_tensor_v3(storage_obj, storage_offset, size, stride, requires_grad, backward_hooks, dtype=None):
    """Reconstruct a tensor from its storage (v3 protocol, PyTorch 2.x)."""
    return _rebuild_tensor_v2(storage_obj, storage_offset, size, stride, requires_grad, backward_hooks)


class _TorchUnpickler(pickle.Unpickler):
    """Custom Unpickler that handles PyTorch-specific pickle instructions."""

    def __init__(self, file, zip_file=None, prefix=None, **kwargs):
        super().__init__(file, **kwargs)
        self.zip_file = zip_file
        self.prefix = prefix

    def find_class(self, module, name):
        # rebuild functions
        if name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if name == "_rebuild_tensor_v3":
            return _rebuild_tensor_v3

        # OrderedDict
        if module == "collections" and name == "OrderedDict":
            return OrderedDict

        # torch.Size -> tuple
        if name == "Size":
            return tuple

        # torch.*Storage classes
        if name in _DTYPE_INFO and "Storage" in name:
            return _StorageTypePlaceholder(name)

        # torch.storage.TypedStorage / torch.UntypedStorage
        if name == "TypedStorage":
            return _StorageTypePlaceholder("TypedStorage")
        if name == "UntypedStorage":
            return _UntypedStoragePlaceholder

        # torch dtype strings that appear as torch.float32 etc.
        if module == "torch" and name in ("float32", "float64", "float16", "bfloat16",
                                           "int64", "int32", "int16", "int8", "uint8"):
            return f"torch.{name}"

        # _rebuild_wrapper_subclass, etc.
        if "_rebuild" in name:
            # Return a generic handler that just passes through
            def _generic_rebuild(*args, **kwargs):
                # Try to find a tensor in args
                for a in args:
                    if isinstance(a, Tensor):
                        return a
                return None
            return _generic_rebuild

        # set_from_file, etc.
        if module == "torch._utils" and name == "_rebuild_parameter":
            def _rebuild_parameter(data, requires_grad, backward_hooks):
                return data
            return _rebuild_parameter

        if module == "torch._utils" and name == "_rebuild_parameter_with_state":
            def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
                return data
            return _rebuild_parameter_with_state

        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            print(f"  WARNING: Unknown class {module}.{name}, returning dummy")
            return lambda *args, **kwargs: args[0] if args else None

    def persistent_load(self, saved_id):
        """Handle persistent_id references for tensor storage data."""
        assert isinstance(saved_id, tuple), f"Unexpected persistent_id format: {saved_id}"
        typename = saved_id[0]

        if typename == "storage":
            # Format: ("storage", storage_type, key, location, numel)
            # or PyTorch 2.x: ("storage", UntypedStorage, key, location, numel)
            storage_type = saved_id[1]
            key = str(saved_id[2])
            location = saved_id[3]
            numel = saved_id[4]

            data_path = f"{self.prefix}/data/{key}"

            # Read raw bytes
            raw = self.zip_file.read(data_path)

            # Determine format from storage_type
            dtype_key = None
            if isinstance(storage_type, _StorageTypePlaceholder):
                dtype_key = storage_type.name
            elif isinstance(storage_type, str):
                dtype_key = storage_type
            elif isinstance(storage_type, type) and hasattr(storage_type, '__name__'):
                dtype_key = storage_type.__name__

            if dtype_key and dtype_key in _DTYPE_INFO:
                fmt_char, itemsize = _DTYPE_INFO[dtype_key]
            elif dtype_key == "TypedStorage" or dtype_key == "UntypedStorage":
                # For untyped storage, figure out from byte count
                # numel is the number of bytes for UntypedStorage
                # We'll figure out the actual dtype when _rebuild_tensor_v3 is called
                # For now, treat as raw bytes - the numel IS the byte count
                # Common case: float32 (4 bytes each)
                byte_count = len(raw)
                if byte_count == numel:
                    # numel = number of bytes -> UntypedStorage
                    # Just store as bytes, will be reinterpreted later
                    # But we need to guess dtype. Check if numel is divisible by 4 (float32)
                    # This is the most common case for neural network weights
                    fmt_char, itemsize = "f", 4
                    numel = byte_count // 4
                elif byte_count == numel * 4:
                    fmt_char, itemsize = "f", 4
                else:
                    fmt_char, itemsize = "f", 4
                    numel = byte_count // 4
            else:
                # Default: float32
                fmt_char, itemsize = "f", 4

            count = numel
            expected_bytes = count * itemsize

            if fmt_char == "e":
                values = list(struct.unpack(f"<{count}e", raw[:expected_bytes]))
                values = [float(v) for v in values]
            elif fmt_char == "q":
                values = list(struct.unpack(f"<{count}q", raw[:expected_bytes]))
                values = [float(v) for v in values]
            elif fmt_char == "i":
                values = list(struct.unpack(f"<{count}i", raw[:expected_bytes]))
                values = [float(v) for v in values]
            elif fmt_char in ("b", "B"):
                values = list(struct.unpack(f"<{count}{fmt_char}", raw[:expected_bytes]))
                values = [float(v) for v in values]
            else:
                values = list(struct.unpack(f"<{count}{fmt_char}", raw[:expected_bytes]))

            return {"type": dtype_key, "key": key, "data": values, "numel": count}

        raise RuntimeError(f"Unknown persistent_id type: {typename}")


def load_pytorch_checkpoint(path: str) -> dict:
    """Load a PyTorch state_dict from a .pth file."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        print(f"  Zip contents ({len(names)} files):")
        for n in names[:10]:
            print(f"    {n}")
        if len(names) > 10:
            print(f"    ... and {len(names) - 10} more")

        # Find the .pkl file to determine the prefix
        pkl_names = [n for n in names if n.endswith(".pkl")]
        assert len(pkl_names) >= 1, f"No .pkl file found in archive. Files: {names}"
        pkl_name = pkl_names[0]

        # Extract prefix: "ddpm_mnist/data.pkl" -> "ddpm_mnist"
        prefix = pkl_name.rsplit("/", 1)[0] if "/" in pkl_name else ""
        print(f"  Detected prefix: '{prefix}'")

        # Check version info
        version_files = [
            f"{prefix}/.format_version",
            f"{prefix}/version",
        ]
        for vf in version_files:
            if vf in names:
                try:
                    content = zf.read(vf).decode("utf-8").strip()
                    print(f"  {vf}: {content}")
                except:
                    pass

        pkl_data = zf.read(pkl_name)
        unpickler = _TorchUnpickler(io.BytesIO(pkl_data), zip_file=zf, prefix=prefix)
        result = unpickler.load()

    return result


# ============================================================
# Part 3: Neural network layers (pure Python)
# ============================================================

def conv2d(input_tensor, weight, bias, padding=0):
    """
    2D convolution (no stride, no dilation, no groups).

    input_tensor: shape (C_in, H, W)
    weight:       shape (C_out, C_in, kH, kW)
    bias:         shape (C_out,) or None

    Returns: shape (C_out, H_out, W_out)
    """
    C_in, H, W = input_tensor.shape
    C_out, C_in_w, kH, kW = weight.shape
    assert C_in == C_in_w, f"Channel mismatch: input has {C_in}, weight expects {C_in_w}"

    H_out = H + 2 * padding - kH + 1
    W_out = W + 2 * padding - kW + 1

    w_data = weight.data
    in_data = input_tensor.data

    # Build padded input as flat array
    pH = H + 2 * padding
    pW = W + 2 * padding
    padded = [0.0] * (C_in * pH * pW)

    for c in range(C_in):
        c_in_off = c * H * W
        c_pad_off = c * pH * pW
        for h in range(H):
            src_start = c_in_off + h * W
            dst_start = c_pad_off + (h + padding) * pW + padding
            padded[dst_start:dst_start + W] = in_data[src_start:src_start + W]

    # Compute output
    out_data = [0.0] * (C_out * H_out * W_out)

    for co in range(C_out):
        co_w_base = co * C_in_w * kH * kW
        co_out_base = co * H_out * W_out
        for oh in range(H_out):
            for ow in range(W_out):
                val = 0.0
                for ci in range(C_in):
                    ci_pad_base = ci * pH * pW
                    ci_w_base = co_w_base + ci * kH * kW
                    for kh in range(kH):
                        pad_row_off = ci_pad_base + (oh + kh) * pW + ow
                        w_row_off = ci_w_base + kh * kW
                        for kw in range(kW):
                            val += padded[pad_row_off + kw] * w_data[w_row_off + kw]
                out_data[co_out_base + oh * W_out + ow] = val

    if bias is not None:
        b_data = bias.data
        for co in range(C_out):
            b = b_data[co]
            base = co * H_out * W_out
            for i in range(H_out * W_out):
                out_data[base + i] += b

    return Tensor(out_data, (C_out, H_out, W_out))


def batchnorm2d_eval(input_tensor, weight, bias, running_mean, running_var, eps=1e-5):
    """
    BatchNorm2d in eval mode for single sample (no batch dim).
    input: (C, H, W) -> output: (C, H, W)
    """
    C, H, W = input_tensor.shape
    out_data = list(input_tensor.data)
    hw = H * W

    gamma = weight.data
    beta = bias.data
    mean = running_mean.data
    var = running_var.data

    for c in range(C):
        scale = gamma[c] / math.sqrt(var[c] + eps)
        shift = beta[c] - scale * mean[c]
        offset = c * hw
        for i in range(hw):
            out_data[offset + i] = out_data[offset + i] * scale + shift

    return Tensor(out_data, input_tensor.shape)


def leaky_relu(input_tensor, negative_slope=0.01):
    out = [x if x >= 0 else x * negative_slope for x in input_tensor.data]
    return Tensor(out, input_tensor.shape)


def forward_block(x, conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var):
    """One block: Conv2d(7x7, pad=3) -> BatchNorm2d -> LeakyReLU."""
    x = conv2d(x, conv_w, conv_b, padding=3)
    x = batchnorm2d_eval(x, bn_w, bn_b, bn_mean, bn_var)
    x = leaky_relu(x)
    return x


def forward_eps_model(x, state_dict):
    """
    Forward pass of DummyEpsModel.

    Architecture:
        7 blocks of Conv2d(7x7, pad=3) + BatchNorm2d + LeakyReLU
        1 final Conv2d(3x3, pad=1)

    Keys pattern:
        eps_model.conv.{0..6}.0.weight / .0.bias       -> Conv
        eps_model.conv.{0..6}.1.weight / .1.bias        -> BN gamma/beta
        eps_model.conv.{0..6}.1.running_mean / .running_var
        eps_model.conv.7.weight / .7.bias               -> final Conv
    """
    for i in range(7):
        prefix = f"eps_model.conv.{i}"
        conv_w = state_dict[f"{prefix}.0.weight"]
        conv_b = state_dict[f"{prefix}.0.bias"]
        bn_w = state_dict[f"{prefix}.1.weight"]
        bn_b = state_dict[f"{prefix}.1.bias"]
        bn_mean = state_dict[f"{prefix}.1.running_mean"]
        bn_var = state_dict[f"{prefix}.1.running_var"]
        x = forward_block(x, conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var)

    final_w = state_dict["eps_model.conv.7.weight"]
    final_b = state_dict["eps_model.conv.7.bias"]
    x = conv2d(x, final_w, final_b, padding=1)
    return x


# ============================================================
# Part 4: DDPM schedules & sampling
# ============================================================

def compute_ddpm_schedules(beta1, beta2, T):
    """Compute the DDPM noise schedule arrays (indexed 0..T)."""
    beta_t = [beta1 + (beta2 - beta1) * i / T for i in range(T + 1)]
    sqrt_beta_t = [math.sqrt(b) for b in beta_t]
    alpha_t = [1.0 - b for b in beta_t]

    log_alpha_t = [math.log(a) for a in alpha_t]
    cumsum = []
    s = 0.0
    for la in log_alpha_t:
        s += la
        cumsum.append(s)
    alphabar_t = [math.exp(c) for c in cumsum]

    sqrtab = [math.sqrt(ab) for ab in alphabar_t]
    oneover_sqrta = [1.0 / math.sqrt(a) for a in alpha_t]
    sqrtmab = [math.sqrt(1.0 - ab) for ab in alphabar_t]
    mab_over_sqrtmab = [(1.0 - a) / sm if sm > 0 else 0.0
                         for a, sm in zip(alpha_t, sqrtmab)]

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }


def ddpm_sample(state_dict, n_sample, size, n_T, betas, rng):
    """
    DDPM sampling (Algorithm 2 from Ho et al.).

    x_T ~ N(0, I)
    For t = T down to 1:
        z ~ N(0, I) if t > 1, else 0
        eps = model(x_t)
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - ((1-alpha_t)/sqrt(1-alphabar_t)) * eps) + sqrt(beta_t) * z
    """
    schedules = compute_ddpm_schedules(betas[0], betas[1], n_T)
    C, H, W = size
    total = C * H * W

    samples = []
    for s_idx in range(n_sample):
        print(f"\n=== Generating sample {s_idx + 1}/{n_sample} ===")
        x = rng.randn(C, H, W)

        for i in range(n_T, 0, -1):
            if i % 100 == 0 or i == n_T or i == 1:
                print(f"  Step t={i}/{n_T} ...")

            if i > 1:
                z = rng.randn(C, H, W)
            else:
                z = tensor_zeros((C, H, W))

            eps = forward_eps_model(x, state_dict)

            coeff1 = schedules["oneover_sqrta"][i]
            coeff2 = schedules["mab_over_sqrtmab"][i]
            coeff3 = schedules["sqrt_beta_t"][i]

            x_data = x.data
            eps_data = eps.data
            z_data = z.data
            new_data = [0.0] * total
            for j in range(total):
                new_data[j] = coeff1 * (x_data[j] - eps_data[j] * coeff2) + coeff3 * z_data[j]
            x = Tensor(new_data, (C, H, W))

        samples.append(x)

    return samples


# ============================================================
# Part 5: Save image as PNG (minimal encoder)
# ============================================================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def samples_to_grid(samples, n_channels, height, width, nrow):
    """Arrange samples into a grid image (pixel values 0-255)."""
    n = len(samples)
    ncol = nrow
    nrows_grid = (n + ncol - 1) // ncol

    pad = 2
    grid_h = nrows_grid * height + (nrows_grid + 1) * pad
    grid_w = ncol * width + (ncol + 1) * pad

    # Initialize with gray background
    grid = [[[128] * n_channels for _ in range(grid_w)] for _ in range(grid_h)]

    for idx, sample in enumerate(samples):
        row = idx // ncol
        col = idx % ncol
        y_off = pad + row * (height + pad)
        x_off = pad + col * (width + pad)

        s_data = sample.data
        for c in range(n_channels):
            for h in range(height):
                for w_idx in range(width):
                    val = s_data[c * height * width + h * width + w_idx]
                    val = clamp(val, 0.0, 1.0)
                    pixel = int(val * 255.0 + 0.5)
                    grid[y_off + h][x_off + w_idx][c] = pixel

    return grid, grid_h, grid_w


def save_png(filepath, grid, grid_h, grid_w, n_channels):
    """Minimal PNG encoder."""

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    if n_channels == 1:
        color_type = 0   # Grayscale
    elif n_channels == 3:
        color_type = 2   # RGB
    elif n_channels == 4:
        color_type = 6   # RGBA
    else:
        raise ValueError(f"Unsupported channels: {n_channels}")

    ihdr_data = struct.pack(">IIBBBBB", grid_w, grid_h, 8, color_type, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    raw = bytearray()
    for y in range(grid_h):
        raw.append(0)  # filter type = None
        for x in range(grid_w):
            for c in range(n_channels):
                raw.append(grid[y][x][c])

    compressed = zlib.compress(bytes(raw))
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")

    signature = b"\x89PNG\r\n\x1a\n"

    with open(filepath, "wb") as f:
        f.write(signature + ihdr + idat + iend)

    print(f"  PNG saved: {filepath} ({grid_w}x{grid_h}, {n_channels}ch)")


# ============================================================
# Part 6: Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pure-Python DDPM sampler")
    parser.add_argument("--checkpoint", type=str, default="./ddpm_mnist.pth")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sample", type=int, default=16)
    parser.add_argument("--output", type=str, default="./contents/ddpm_sample_from_pureddpm.png")
    args = parser.parse_args()

    # ------- Load -------
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = load_pytorch_checkpoint(args.checkpoint)

    print(f"\nLoaded {len(state_dict)} keys:")
    for k, v in state_dict.items():
        if isinstance(v, Tensor):
            print(f"  {k}: Tensor shape={v.shape}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    # ------- Sample -------
    rng = PureRandom(args.seed)

    n_T = 1000
    betas = (1e-4, 0.02)
    size = (1, 28, 28)  # MNIST: 1 channel, 28x28

    print(f"\nSampling {args.n_sample} image(s)...")
    print("NOTE: Pure Python convolution is very slow.")
    print("      Each denoising step needs 8 convolutions on 28x28.")
    print("      1000 steps x 8 convs = 8000 convolutions per image.")
    print("      Consider using --n-sample 1 for testing.\n")

    samples = ddpm_sample(state_dict, args.n_sample, size, n_T, betas, rng)

    # ------- Save -------
    nrow = int(math.ceil(math.sqrt(args.n_sample)))
    if nrow < 1:
        nrow = 1
    grid, grid_h, grid_w = samples_to_grid(samples, size[0], size[1], size[2], nrow)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_png(args.output, grid, grid_h, grid_w, size[0])
    print(f"\nDone! Output: {args.output}")


if __name__ == "__main__":
    main()