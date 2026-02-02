import argparse
import json
import os
from typing import Any, Dict

import torch
from torchvision.utils import save_image

from mindiffusion.ddpm import DDPM
from mindiffusion.unet import NaiveUnet


def tensor_to_json(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "data": tensor.cpu().tolist(),
    }


def state_dict_to_json(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return {key: tensor_to_json(value) for key, value in state_dict.items()}


def build_ddpm() -> DDPM:
    return DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ddpm_cifar.pth to JSON and run a sample inference."
    )
    parser.add_argument(
        "--pth",
        default="ddpm_cifar.pth",
        help="Path to ddpm_cifar.pth checkpoint.",
    )
    parser.add_argument(
        "--json",
        default="ddpm_cifar.json",
        help="Output JSON file for converted weights.",
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
        default="contents/ddpm_sample_from_pth.png",
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
    state = torch.load(args.pth, map_location="cpu")
    ddpm.load_state_dict(state)

    json_payload = state_dict_to_json(state)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(json_payload, f)

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
