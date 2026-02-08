import os
from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


def export_mnist(out_dir="./dataset_mnist"):
    out_dir = Path(out_dir)

    # 保存为 tensor（0~1），方便直接 save_image
    tf = transforms.ToTensor()

    print("Loading MNIST dataset ...")

    train_ds = MNIST(
        root="./mnist_raw",
        train=True,
        download=True,
        transform=tf,
    )

    test_ds = MNIST(
        root="./mnist_raw",
        train=False,
        download=True,
        transform=tf,
    )

    print(f"Train size: {len(train_ds)}")
    print(f"Test  size: {len(test_ds)}")

    # 创建目录
    for split in ["train", "test"]:
        for label in range(10):
            (out_dir / split / str(label)).mkdir(parents=True, exist_ok=True)

    # 导出函数
    def dump_dataset(ds, split):
        counter = [0] * 10

        for img, label in tqdm(ds, desc=f"Export {split}"):
            idx = counter[label]
            save_path = out_dir / split / str(label) / f"{idx}.png"

            save_image(img, save_path)

            counter[label] += 1

    dump_dataset(train_ds, "train")
    dump_dataset(test_ds, "test")

    print("Done!")
    print(f"Exported to: {out_dir.resolve()}")


if __name__ == "__main__":
    export_mnist()
