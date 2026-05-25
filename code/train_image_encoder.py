# -*- coding: utf-8 -*-
import logging
import os
import random
import time

import Config_SSL as config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from Load_Dataset_val_SSL import ImageToImage2D_val, RandomGenerator, ValGenerator
from nets.unet_blocks import ConvBatchNorm, DownBlock, UpBlock
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils_train import (
    CosineAnnealingWarmRestarts,
    WeightedDiceBCE,
    dice_on_batch,
    iou_on_batch,
    save_on_batch,
)


# --- UNet Model Definition (Image Encoder part of EviVLM) ---
class ImageEncoderUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(ImageEncoderUNet, self).__init__()
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1))
        self.last_activation = nn.Sigmoid() if n_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return self.last_activation(logits)


# --- Logging Configuration ---
def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding="UTF-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path, model_name):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state["epoch"]
    best_model = state["best_model"]

    if best_model:
        filename = os.path.join(save_path, f"best_model-{model_name}.pth.tar")
    else:
        filename = os.path.join(save_path, f"model-{model_name}-{epoch:02d}.pth.tar")

    torch.save(state, filename)
    logging.info(f"\t Model saved to {filename}")


def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)


def train_one_epoch(
    loader,
    model,
    criterion,
    optimizer,
    writer,
    epoch,
    lr_scheduler,
    logger,
    scaler,
    device,
):
    model.train()
    end = time.time()
    loss_sum, dice_sum, iou_sum = 0, 0, 0

    # Enable AMP if desired, but default to False for stability on GTX 1650 during debugging
    use_amp = False

    for i, (sampled_batch, names) in enumerate(loader, 1):
        images, masks = sampled_batch["image"], sampled_batch["label"]
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images)
            loss = criterion(preds, masks.half() if use_amp else masks.float())

        if torch.isnan(loss):
            logger.error(
                f"NaN loss detected at Epoch {epoch + 1}, batch {i}. Skipping..."
            )
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Calculate metrics using float32 for stability
        with torch.no_grad():
            preds_eval = preds.float()
            train_dice = criterion._show_dice(preds_eval, masks.float())
            train_iou = iou_on_batch(masks, preds_eval)

        batch_size = images.size(0)
        loss_sum += loss.item() * batch_size
        dice_sum += train_dice * batch_size
        iou_sum += train_iou * batch_size

        if i % config.print_frequency == 0:
            avg_loss = loss_sum / (i * loader.batch_size)
            avg_dice = dice_sum / (i * loader.batch_size)
            avg_iou = iou_sum / (i * loader.batch_size)
            logger.info(
                f"Train Epoch: [{epoch + 1}][{i}/{len(loader)}] Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) Dice: {train_dice:.4f} (Avg: {avg_dice:.4f}) IoU: {train_iou:.4f} (Avg: {avg_iou:.4f})"
            )

    if lr_scheduler is not None:
        lr_scheduler.step()


def validate(loader, model, criterion, epoch, logger, device, visualize_path):
    model.eval()
    loss_sum, dice_sum, iou_sum = 0, 0, 0

    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(loader, 1):
            images, masks = sampled_batch["image"], sampled_batch["label"]
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks.half())

            val_dice = criterion._show_dice(preds, masks.half())
            val_iou = iou_on_batch(masks, preds)

            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            dice_sum += val_dice * batch_size
            iou_sum += val_iou * batch_size

            if epoch % config.vis_frequency == 0:
                vis_path = os.path.join(visualize_path, str(epoch + 1))
                if not os.path.isdir(vis_path):
                    os.makedirs(vis_path)
                save_on_batch(images, masks, preds.float(), names, vis_path + "/")

    avg_loss = loss_sum / len(loader.dataset)
    avg_dice = dice_sum / len(loader.dataset)
    avg_iou = iou_sum / len(loader.dataset)

    logger.info(
        f"Validation Epoch: [{epoch + 1}] Avg Loss: {avg_loss:.4f} Avg Dice: {avg_dice:.4f} Avg IoU: {avg_iou:.4f}"
    )
    return avg_loss, avg_dice


def main():
    # Setup paths
    task_name = "ImageEncoder_Pretrain"
    model_name = "UNet"
    session_name = "Pretrain_" + time.strftime("%m.%d_%Hh%M")
    save_path = os.path.join(task_name, model_name, session_name)
    model_path = os.path.join(save_path, "models")
    log_path = os.path.join(save_path, session_name + ".log")
    visualize_path = os.path.join(save_path, "visualize_val")

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    logger = logger_config(log_path)

    # Dataset setup
    # Note: Using absolute paths or relative to project root.
    # Based on session_context, project root is D:\VLM_Medical_Imaging
    train_dataset_path = "D:/VLM_Medical_Imaging/dataset/Train_Folder/"
    val_dataset_path = "D:/VLM_Medical_Imaging/dataset/Val_Folder/"

    train_tf = transforms.Compose(
        [RandomGenerator(output_size=[config.img_size, config.img_size])]
    )
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    train_dataset = ImageToImage2D_val(
        train_dataset_path, "Pretrain", train_tf, image_size=config.img_size
    )
    val_dataset = ImageToImage2D_val(
        val_dataset_path, "Pretrain", val_tf, image_size=config.img_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_val,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Model, Optimizer, Loss
    device = torch.device(config.device)
    model = ImageEncoderUNet(
        n_channels=config.n_channels, n_classes=config.n_labels
    ).to(device)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-4
        )
    else:
        lr_scheduler = None

    writer = SummaryWriter(os.path.join(save_path, "tensorboard_logs"))

    max_dice = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        logger.info(f"\n========= Epoch [{epoch + 1}/{config.epochs}] =========")

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            writer,
            epoch,
            lr_scheduler,
            logger,
            scaler,
            device,
        )
        val_loss, val_dice = validate(
            val_loader, model, criterion, epoch, logger, device, visualize_path
        )

        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Dice", val_dice, epoch)

        if val_dice > max_dice:
            logger.info(f"\t Best dice improved from {max_dice:.4f} to {val_dice:.4f}")
            max_dice = val_dice
            best_epoch = epoch + 1
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "best_model": True,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_path,
                model_name,
            )

        if (epoch + 1) - best_epoch > config.early_stopping_patience:
            logger.info("\t Early stopping triggered!")
            break


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    main()
