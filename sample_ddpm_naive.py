import argparse
import json
import math
import os
import random
import struct
import zlib
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def reshape_flat(flat: Sequence[float], shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return list(flat[: shape[0]])
    step = 1
    for dim in shape[1:]:
        step *= dim
    return [reshape_flat(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def load_state_dict_from_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: reshape_flat(value["data"], value["shape"]) for key, value in raw.items()}


def zeros(shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    return [zeros(shape[1:]) for _ in range(shape[0])]


def randn(shape: Sequence[int], rng: random.Random) -> Any:
    if len(shape) == 1:
        return [rng.gauss(0.0, 1.0) for _ in range(shape[0])]
    return [randn(shape[1:], rng) for _ in range(shape[0])]


def tensor_shape(x: Any) -> Tuple[int, ...]:
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        if not x:
            break
        x = x[0]
    return tuple(shape)


def add_tensors(a: Any, b: Any) -> Any:
    shape_a = tensor_shape(a)
    shape_b = tensor_shape(b)
    if shape_a == shape_b:
        if isinstance(a, list):
            return [add_tensors(ai, bi) for ai, bi in zip(a, b)]
        return a + b
    if shape_b[-2:] == (1, 1) and shape_a[:-2] == shape_b[:-2]:
        if isinstance(a, list):
            return [add_tensors(ai, b[i]) for i, ai in enumerate(a)]
        return a + b
    if shape_b[-2:] == (1, 1) and len(shape_b) == 4 and len(shape_a) == 4:
        out = zeros(shape_a)
        for bi in range(shape_a[0]):
            for ci in range(shape_a[1]):
                offset = b[bi][ci][0][0]
                for hi in range(shape_a[2]):
                    for wi in range(shape_a[3]):
                        out[bi][ci][hi][wi] = a[bi][ci][hi][wi] + offset
        return out
    raise ValueError(f"Unsupported broadcast shapes {shape_a} and {shape_b}")


def mul_scalar(a: Any, scalar: float) -> Any:
    if isinstance(a, list):
        return [mul_scalar(ai, scalar) for ai in a]
    return a * scalar


def sub_tensors(a: Any, b: Any) -> Any:
    if isinstance(a, list):
        return [sub_tensors(ai, bi) for ai, bi in zip(a, b)]
    return a - b


def relu(x: Any) -> Any:
    if isinstance(x, list):
        return [relu(xi) for xi in x]
    return x if x > 0.0 else 0.0


def conv2d(x: List[List[List[List[float]]]], weight: Any, bias: Any, padding: int = 1) -> Any:
    batch, in_channels, height, width = tensor_shape(x)
    out_channels = len(weight)
    kernel_h = len(weight[0][0])
    kernel_w = len(weight[0][0][0])
    out = zeros((batch, out_channels, height, width))
    for b in range(batch):
        for oc in range(out_channels):
            bias_val = bias[oc] if bias is not None else 0.0
            for h in range(height):
                for w in range(width):
                    acc = bias_val
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            ih = h + kh - padding
                            if ih < 0 or ih >= height:
                                continue
                            for kw in range(kernel_w):
                                iw = w + kw - padding
                                if iw < 0 or iw >= width:
                                    continue
                                acc += x[b][ic][ih][iw] * weight[oc][ic][kh][kw]
                    out[b][oc][h][w] = acc
    return out


def conv_transpose2d(
    x: List[List[List[List[float]]]],
    weight: Any,
    bias: Any,
    stride: int,
) -> Any:
    batch, in_channels, height, width = tensor_shape(x)
    out_channels = len(weight[0])
    kernel_h = len(weight[0][0])
    kernel_w = len(weight[0][0][0])
    out_height = (height - 1) * stride + kernel_h
    out_width = (width - 1) * stride + kernel_w
    out = zeros((batch, out_channels, out_height, out_width))
    for b in range(batch):
        for ic in range(in_channels):
            for h in range(height):
                for w in range(width):
                    val = x[b][ic][h][w]
                    base_h = h * stride
                    base_w = w * stride
                    for oc in range(out_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                out[b][oc][base_h + kh][base_w + kw] += (
                                    val * weight[ic][oc][kh][kw]
                                )
        if bias is not None:
            for oc in range(out_channels):
                bias_val = bias[oc]
                for h in range(out_height):
                    for w in range(out_width):
                        out[b][oc][h][w] += bias_val
    return out


def group_norm(x: Any, weight: Any, bias: Any, num_groups: int = 8, eps: float = 1e-5) -> Any:
    batch, channels, height, width = tensor_shape(x)
    group_size = channels // num_groups
    out = zeros((batch, channels, height, width))
    for b in range(batch):
        for g in range(num_groups):
            c_start = g * group_size
            c_end = c_start + group_size
            values = []
            for c in range(c_start, c_end):
                for h in range(height):
                    for w in range(width):
                        values.append(x[b][c][h][w])
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / len(values)
            inv_std = 1.0 / math.sqrt(var + eps)
            for c in range(c_start, c_end):
                w_scale = weight[c] if weight is not None else 1.0
                b_shift = bias[c] if bias is not None else 0.0
                for h in range(height):
                    for w in range(width):
                        normalized = (x[b][c][h][w] - mean) * inv_std
                        out[b][c][h][w] = normalized * w_scale + b_shift
    return out


def max_pool2d(x: Any, kernel: int = 2, stride: int = 2) -> Any:
    batch, channels, height, width = tensor_shape(x)
    out_height = height // stride
    out_width = width // stride
    out = zeros((batch, channels, out_height, out_width))
    for b in range(batch):
        for c in range(channels):
            for h in range(out_height):
                for w in range(out_width):
                    base_h = h * stride
                    base_w = w * stride
                    max_val = x[b][c][base_h][base_w]
                    for kh in range(kernel):
                        for kw in range(kernel):
                            max_val = max(max_val, x[b][c][base_h + kh][base_w + kw])
                    out[b][c][h][w] = max_val
    return out


def avg_pool2d(x: Any, kernel: int = 4, stride: int = 4) -> Any:
    batch, channels, height, width = tensor_shape(x)
    out_height = height // stride
    out_width = width // stride
    out = zeros((batch, channels, out_height, out_width))
    area = kernel * kernel
    for b in range(batch):
        for c in range(channels):
            for h in range(out_height):
                for w in range(out_width):
                    base_h = h * stride
                    base_w = w * stride
                    total = 0.0
                    for kh in range(kernel):
                        for kw in range(kernel):
                            total += x[b][c][base_h + kh][base_w + kw]
                    out[b][c][h][w] = total / area
    return out


def concat_channels(a: Any, b: Any) -> Any:
    out = []
    for ai, bi in zip(a, b):
        out.append(ai + bi)
    return out


def linear(x: List[List[float]], weight: Any, bias: Any) -> List[List[float]]:
    out = []
    for row in x:
        out_row = []
        for w_row, b_val in zip(weight, bias or [0.0 for _ in range(len(weight))]):
            acc = 0.0
            for v, w_val in zip(row, w_row):
                acc += v * w_val
            out_row.append(acc + b_val)
        out.append(out_row)
    return out


def time_siren(t: List[float], weight1: Any, weight2: Any, bias2: Any) -> Any:
    x = [[value] for value in t]
    lin1 = linear(x, weight1, None)
    sin1 = [[math.sin(v) for v in row] for row in lin1]
    return linear(sin1, weight2, bias2)


class NaiveUnetPure:
    def __init__(self, params: Dict[str, Any], n_feat: int = 128) -> None:
        self.params = params
        self.n_feat = n_feat

    def conv3(self, x: Any, prefix: str, is_res: bool = False) -> Any:
        main = conv2d(
            x,
            self.params[f"{prefix}.main.0.weight"],
            self.params.get(f"{prefix}.main.0.bias"),
        )
        main = group_norm(
            main,
            self.params.get(f"{prefix}.main.1.weight"),
            self.params.get(f"{prefix}.main.1.bias"),
        )
        main = relu(main)

        conv = conv2d(
            main,
            self.params[f"{prefix}.conv.0.weight"],
            self.params.get(f"{prefix}.conv.0.bias"),
        )
        conv = group_norm(
            conv,
            self.params.get(f"{prefix}.conv.1.weight"),
            self.params.get(f"{prefix}.conv.1.bias"),
        )
        conv = relu(conv)
        conv = conv2d(
            conv,
            self.params[f"{prefix}.conv.3.weight"],
            self.params.get(f"{prefix}.conv.3.bias"),
        )
        conv = group_norm(
            conv,
            self.params.get(f"{prefix}.conv.4.weight"),
            self.params.get(f"{prefix}.conv.4.bias"),
        )
        conv = relu(conv)

        if is_res:
            summed = add_tensors(main, conv)
            return mul_scalar(summed, 1.0 / 1.414)
        return conv

    def unet_down(self, x: Any, prefix: str) -> Any:
        out = self.conv3(x, f"{prefix}.model.0")
        return max_pool2d(out)

    def unet_up(self, x: Any, skip: Any, prefix: str) -> Any:
        merged = concat_channels(x, skip)
        out = conv_transpose2d(
            merged,
            self.params[f"{prefix}.model.0.weight"],
            self.params.get(f"{prefix}.model.0.bias"),
            stride=2,
        )
        out = self.conv3(out, f"{prefix}.model.1")
        out = self.conv3(out, f"{prefix}.model.2")
        return out

    def forward(self, x: Any, t: List[float]) -> Any:
        x = self.conv3(x, "init_conv", is_res=True)

        down1 = self.unet_down(x, "down1")
        down2 = self.unet_down(down1, "down2")
        down3 = self.unet_down(down2, "down3")

        thro = avg_pool2d(down3)
        thro = relu(thro)

        temb = time_siren(
            t,
            self.params["timeembed.lin1.weight"],
            self.params["timeembed.lin2.weight"],
            self.params.get("timeembed.lin2.bias"),
        )
        temb = [[[[value]] for value in row] for row in temb]

        thro = add_tensors(thro, temb)
        thro = conv_transpose2d(
            thro,
            self.params["up0.0.weight"],
            self.params.get("up0.0.bias"),
            stride=4,
        )
        thro = group_norm(
            thro,
            self.params.get("up0.1.weight"),
            self.params.get("up0.1.bias"),
        )
        thro = relu(thro)

        up1 = self.unet_up(thro, down3, "up1")
        up1 = add_tensors(up1, temb)
        up2 = self.unet_up(up1, down2, "up2")
        up3 = self.unet_up(up2, down1, "up3")

        out = concat_channels(up3, x)
        out = conv2d(
            out,
            self.params["out.weight"],
            self.params.get("out.bias"),
        )
        return out


def ddpm_schedules(beta1: float, beta2: float, steps: int) -> Dict[str, List[float]]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    beta_t = [beta1 + (beta2 - beta1) * i / steps for i in range(steps + 1)]
    sqrt_beta_t = [math.sqrt(b) for b in beta_t]
    alpha_t = [1.0 - b for b in beta_t]
    log_alpha_t = [math.log(a) for a in alpha_t]
    alphabar_t = []
    running = 0.0
    for value in log_alpha_t:
        running += value
        alphabar_t.append(math.exp(running))

    sqrtab = [math.sqrt(a) for a in alphabar_t]
    oneover_sqrta = [1.0 / math.sqrt(a) for a in alpha_t]
    sqrtmab = [math.sqrt(1.0 - a) for a in alphabar_t]
    mab_over_sqrtmab = [(1.0 - a) / smab for a, smab in zip(alpha_t, sqrtmab)]

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab,
    }


def sample_ddpm(
    eps_model: NaiveUnetPure,
    schedules: Dict[str, List[float]],
    n_samples: int,
    size: Tuple[int, int, int],
    rng: random.Random,
) -> Any:
    channels, height, width = size
    x = randn((n_samples, channels, height, width), rng)
    n_T = len(schedules["alpha_t"]) - 1
    for i in range(n_T, 0, -1):
        z = randn((n_samples, channels, height, width), rng) if i > 1 else zeros((n_samples, channels, height, width))
        t = [i / n_T for _ in range(n_samples)]
        eps = eps_model.forward(x, t)
        scaled = sub_tensors(x, mul_scalar(eps, schedules["mab_over_sqrtmab"][i]))
        scaled = mul_scalar(scaled, schedules["oneover_sqrta"][i])
        noise = mul_scalar(z, schedules["sqrt_beta_t"][i])
        x = add_tensors(scaled, noise)
    return x


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def tensor_to_image(sample: Any) -> List[List[List[int]]]:
    channels = len(sample)
    height = len(sample[0])
    width = len(sample[0][0])
    image = []
    for h in range(height):
        row = []
        for w in range(width):
            pixel = []
            for c in range(channels):
                value = (sample[c][h][w] + 1.0) * 0.5
                value = clamp(value, 0.0, 1.0)
                pixel.append(int(round(value * 255.0)))
            row.append(pixel)
        image.append(row)
    return image


def make_grid(samples: Any, grid_size: int) -> List[List[List[int]]]:
    num_samples = len(samples)
    sample_image = tensor_to_image(samples[0])
    height = len(sample_image)
    width = len(sample_image[0])
    canvas = [
        [[0, 0, 0] for _ in range(width * grid_size)]
        for _ in range(height * grid_size)
    ]
    for idx, sample in enumerate(samples):
        row = idx // grid_size
        col = idx % grid_size
        image = tensor_to_image(sample)
        for h in range(height):
            for w in range(width):
                canvas[row * height + h][col * width + w] = image[h][w]
    return canvas


def write_png(path: str, image: List[List[List[int]]]) -> None:
    height = len(image)
    width = len(image[0])
    raw = bytearray()
    for row in image:
        raw.append(0)
        for pixel in row:
            raw.extend(pixel)
    compressor = zlib.compressobj()
    compressed = compressor.compress(bytes(raw)) + compressor.flush()

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def build_eps_model(state_dict: Dict[str, Any]) -> NaiveUnetPure:
    params = {}
    for key, value in state_dict.items():
        if key.startswith("eps_model."):
            params[key[len("eps_model.") :]] = value
    return NaiveUnetPure(params=params, n_feat=128)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pure Python DDPM sampler (no torch) for JSON weights."
    )
    parser.add_argument(
        "--json",
        default="ddpm_cifar.json",
        help="Path to JSON weights (converted from ddpm_cifar.pth).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--out",
        default="contents/ddpm_sample_from_naiveddpm.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate.",
    )
    args = parser.parse_args()

    state_dict = load_state_dict_from_json(args.json)
    eps_model = build_eps_model(state_dict)
    schedules = ddpm_schedules(1e-4, 0.02, 1000)

    rng = random.Random(args.seed)
    samples = sample_ddpm(eps_model, schedules, args.num_samples, (3, 32, 32), rng)

    grid_size = math.ceil(math.sqrt(args.num_samples))
    image = make_grid(samples, grid_size)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_png(args.out, image)


if __name__ == "__main__":
    main()
