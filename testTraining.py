

import os
os.environ["WANDB_DIR"] = "./wandb_logs/"
import pathlib
import warnings
import random
import time
import gc
from typing import Tuple
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from batch_finder import optimum_batch_size
from config import set_seed, device
from data_loader import (
    TRAIN_DIR,
    VALID_DIR,
    augment_and_save,
    data_distribution,
    imagenet_stats,
    img_transforms,
    plot_data_distribution,
    show_data,
)
from metrics import Metrics
from model import FireFinder
from trainer import Trainer
from lr_finder import LearningRateFinder
from torch import optim

from sklearn.metrics import confusion_matrix
import seaborn as sns

# hyper params
EPOCHS = 51  # 100
DROPOUT = .6
LR = 2.14e-4  # 1e-6
BATCH_SIZE = 32
BACKBONE = "resnet18"
cuda = torch.device("cuda")

def is_valid_file(path):
    return path.endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')) and not path.startswith('.ipynb_checkpoints')

def create_dataloader(directory: str, batch_size: int, shuffle: bool = False, transform=None) -> DataLoader:
    data = datasets.ImageFolder(directory, transform=transform, is_valid_file=is_valid_file)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def setup_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader]:
    return create_dataloader(
        TRAIN_DIR, config["batch_size"], shuffle=True, transform=img_transforms["train"]
    ), create_dataloader(
        VALID_DIR, config["batch_size"], transform=img_transforms["valid"]
    )

def find_lr(model: FireFinder, optimizer: optim.Adam, dataloader: DataLoader) -> float:
    lr_finder = LearningRateFinder(model, optimizer, device)
    best_lr = lr_finder.lr_range_test(dataloader, start_lr=1e-2, end_lr=1e-5)
    return best_lr

def train(model: FireFinder, trainer: Trainer, config: dict):
    train_dataloader, valid_dataloader = setup_dataloaders(config)
    print("훈련 데이터")
    plot_data_distribution(data_distribution(train_dataloader.dataset, TRAIN_DIR))
    print("\n검증 데이터")
    plot_data_distribution(data_distribution(valid_dataloader.dataset, VALID_DIR))
    print(f"______________")
    start = time.time()

    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    example_images = []

    wandb.watch(model, log="all", log_freq=10)
    model.classes_ = ['Fire', 'NoFire']

    for epoch in range(config["epochs"]):
        train_loss, train_acc, train_preds, train_labels = trainer.train_one_epoch(train_dataloader)
        valid_loss, valid_acc, valid_preds, valid_labels = trainer.validate_one_epoch(valid_dataloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        log_metrics_and_visualize(epoch, train_loss, valid_loss, train_acc, valid_acc,
                                  valid_dataloader, model, example_images, valid_preds, valid_labels, config)

        print(
            f"\nEpoch: {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_acc:.4f}"
        )

    print(f"Time elapsed: {time.time() - start} seconds.")

    return train_losses, valid_losses, train_accs, valid_accs


def log_metrics_and_visualize(epoch, train_loss, valid_loss, train_acc, valid_acc,
                              valid_dataloader, model, example_images, valid_preds, valid_labels, config):
    device = cuda
    step = epoch  # 스텝 값을 epoch으로 설정

    # 훈련 및 검증 메트릭을 로그에 기록
    wandb.log({
        "train_acc": 100. * train_acc,
        "train_loss": train_loss,
        "epoch": epoch
    })
    wandb.log({
        "val_acc": 100. * valid_acc,
        "val_loss": valid_loss,
        "epoch": epoch
    })


    # 예시 이미지를 로그할 때 텐서를 .cpu()로 변환 후 numpy로 변환
    if epoch == 0:
        data, target = next(iter(valid_dataloader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)

        # 예시 이미지를 로그할 때 텐서를 .cpu()로 변환 후 numpy로 변환
        example_images.append(wandb.Image(
            data[0].cpu().numpy().transpose(1, 2, 0), caption="Pred: {} Truth: {}".format(pred[0].item(), target[0].item())))

    # 예시 이미지 로그
    if example_images:
        wandb.log({"examples": example_images, "epoch": epoch})

def main(
    aug_data: bool = False,
    find_batch: bool = False,
    find_lr_rate: bool = False,
    use_wandb: bool = True,
    use_ipex=True,
):
    set_seed(42)
    print(f"Device {device}")
    print(f"Train folder {TRAIN_DIR}")
    print(f"Validation folder {VALID_DIR}")
    print(f"Using epoch: {EPOCHS}")
    print(f"Using Dropout: {DROPOUT}")

    batch_size = BATCH_SIZE

    if aug_data:
        print("Augmenting training and validation datasets...")
        t1 = time.time()
        augment_and_save(TRAIN_DIR)
        augment_and_save(VALID_DIR)
        print(f"Done Augmenting in {time.time() - t1} seconds...")

    model = FireFinder(simple=True, dropout=DROPOUT, backbone=BACKBONE)  # 수동으로 백본 선택
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if find_batch:
        print(f"Finding optimum batch size...")
        batch_size = optimum_batch_size(model, input_size=(3, 224, 224))
    print(f"Using batch size: {batch_size}")

    best_lr = LR
    if find_lr_rate:
        print("Finding best init lr...")
        train_dataloader = create_dataloader(
            TRAIN_DIR,
            batch_size=batch_size,
            shuffle=True,
            transform=img_transforms["train"],
        )
        best_lr = find_lr(model, optimizer, train_dataloader)
        del model, optimizer
        gc.collect()
        if device == torch.device("cuda"):
            torch.xpu.empty_cache()
    print(f"Using learning rate: {best_lr}")

    if use_wandb:
        wandb.init(project="FireFinder", config={
            "learning_rate": best_lr,
            "epochs": EPOCHS,
            "batch_size": batch_size,
            "backbone": BACKBONE,  # 로그에 백본 기록
        })

    trainer = Trainer(
        model=model,
        optimizer=optim.Adam,
        lr=best_lr,
        epochs=EPOCHS,
        device=cuda,
        use_ipex=use_ipex,
    )

    config={"lr": best_lr, "batch_size": batch_size, "epochs": EPOCHS}

    train(model, trainer, config)

if __name__ == "__main__":
    main(
        aug_data=False, find_batch=False, find_lr_rate=False, use_wandb=True, use_ipex=False
    )
