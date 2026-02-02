import argparse
import json
import os
from typing import Any, Dict

import torch
from torchvision.utils import save_image

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet


def json_to_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["data"], dtype=torch.float32).reshape(payload["shape"])


def load_state_dict_from_json(path: str) -> Dict[str, torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {key: json_to_tensor(value) for key, value in raw.items()}


def build_ddpm() -> DDPM:
    return DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load DDPM weights from JSON and sample unconditionally."
    )
    parser.add_argument(
        "--json",
        default="ddpm_cifar.json",
        help="Path to JSON weights (converted from ddpm_cifar.pth).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference, e.g. cpu, cuda, mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--out",
        default="contents/ddpm_sample_from_json.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate.",
    )
    args = parser.parse_args()

    ddpm = build_ddpm()
    state_dict = load_state_dict_from_json(args.json)
    ddpm.load_state_dict(state_dict)

    device = torch.device(args.device)
    ddpm.to(device)
    ddpm.eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with torch.no_grad():
        samples = ddpm.sample(args.num_samples, (3, 32, 32), device)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image(samples, args.out, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
