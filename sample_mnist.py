#!/usr/bin/env python3
"""
Sample MNIST images from a trained DDPM model.

Example:
    python sample_mnist.py --seed 123 --n-sample 16 --device cuda
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from mindiffusion.mnist_ddpm import DDPM, DummyEpsModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MNIST images from DDPM.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("./ddpm_mnist.pth"),
        help="Path to the trained DDPM checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=16,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./contents/ddpm_sample_mnist.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    ddpm = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ddpm.load_state_dict(checkpoint)
    ddpm.to(device)
    ddpm.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        samples = ddpm.sample(args.n_sample, (1, 28, 28), device)
        grid = make_grid(samples, nrow=int(args.n_sample**0.5))
        save_image(grid, args.output)


if __name__ == "__main__":
    main()
