import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def export_cifar10(root_dir="./dataset_cifar10"):
    # CIFAR-10 的类别名称（官方顺序）
    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # 创建基础目录
    for split in ["train", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

    # CIFAR-10 本身已经是 PIL Image，不需要 transform
    train_set = CIFAR10(
        root="./cifar10_raw",
        train=True,
        download=True
    )
    test_set = CIFAR10(
        root="./cifar10_raw",
        train=False,
        download=True
    )

    def save_split(dataset, split_name):
        print(f"Exporting {split_name} set...")
        for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
            cls_name = class_names[label]
            save_dir = os.path.join(root_dir, split_name, cls_name)
            save_path = os.path.join(save_dir, f"{idx:05d}.png")
            img.save(save_path)

    save_split(train_set, "train")
    save_split(test_set, "test")

    print("✅ CIFAR-10 export finished!")

if __name__ == "__main__":
    export_cifar10()
