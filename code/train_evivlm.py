# -*- coding: utf-8 -*-
"""
Train EviVLM (vision + text) on image + mask + report (Excel) dataset.

This script is a minimal adaptation of `train_image_encoder.py` to use the
multimodal `EviVLM` model and include the cross-modal similarity loss.

It does NOT start training automatically when added — run manually.
"""
import logging
import os
import random
import time

import Config_SSL as config
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from Load_Dataset_val_SSL import ImageToImage2D_val, ImageToImage2D, RandomGenerator, ValGenerator
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from nets.EviVLM import EviVLM
from utils_train import (
    CosineAnnealingWarmRestarts,
    WeightedDiceBCE,
    dice_on_batch,
    iou_on_batch,
    save_on_batch,
)


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
    lambda_sim=1.0,
):
    model.train()
    loss_sum, dice_sum, iou_sum = 0, 0, 0

    for i, (sampled_batch, names) in enumerate(loader, 1):
        images, masks = sampled_batch["image"], sampled_batch["label"]
        texts = sampled_batch.get("text", [""] * images.size(0))
        images, masks = images.to(device), masks.to(device)

        # texts is a list of strings (tokenizer inside model handles device placement)

        optimizer.zero_grad()
        # Keep the multimodal forward pass in float32 for stability.
        prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model(images, texts)
        seg_loss = criterion(prob_VL, masks.float())
        total_loss = seg_loss + lambda_sim * loss_sim

        if torch.isnan(total_loss):
            logger.error(f"NaN loss detected at Epoch {epoch + 1}, batch {i}. Skipping...")
            continue

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds_eval = prob_VL.float()
            train_dice = criterion._show_dice(preds_eval, masks.float())
            train_iou = iou_on_batch(masks, preds_eval)

        batch_size = images.size(0)
        loss_sum += total_loss.item() * batch_size
        dice_sum += train_dice * batch_size
        iou_sum += train_iou * batch_size

        if i % config.print_frequency == 0:
            avg_loss = loss_sum / (i * loader.batch_size)
            avg_dice = dice_sum / (i * loader.batch_size)
            avg_iou = iou_sum / (i * loader.batch_size)
            logger.info(
                f"Train Epoch: [{epoch + 1}][{i}/{len(loader)}] Total Loss: {total_loss.item():.4f} (Avg: {avg_loss:.4f}) Dice: {train_dice:.4f} (Avg: {avg_dice:.4f}) IoU: {train_iou:.4f} (Avg: {avg_iou:.4f})"
            )

    if lr_scheduler is not None:
        lr_scheduler.step()


def validate(loader, model, criterion, epoch, logger, device, visualize_path):
    model.eval()
    loss_sum, dice_sum, iou_sum = 0, 0, 0

    with torch.no_grad():
        for i, (sampled_batch, names) in enumerate(loader, 1):
            images, masks = sampled_batch["image"], sampled_batch["label"]
            texts = sampled_batch.get("text", [""] * images.size(0))
            images, masks = images.to(device), masks.to(device)

            prob_V, prob_L, prob_VL, evi_V, evi_L, evi_VL, loss_sim = model(images, texts)
            seg_loss = criterion(prob_VL, masks.float())

            val_dice = criterion._show_dice(prob_VL, masks.float())
            val_iou = iou_on_batch(masks, prob_VL)

            batch_size = images.size(0)
            loss_sum += seg_loss.item() * batch_size
            dice_sum += val_dice * batch_size
            iou_sum += val_iou * batch_size

            if epoch % config.vis_frequency == 0:
                vis_path = os.path.join(visualize_path, str(epoch + 1))
                if not os.path.isdir(vis_path):
                    os.makedirs(vis_path)
                save_on_batch(images, masks, prob_VL.float(), names, vis_path + "/")

    avg_loss = loss_sum / len(loader.dataset)
    avg_dice = dice_sum / len(loader.dataset)
    avg_iou = iou_sum / len(loader.dataset)
    logger.info(
        f"Validation Epoch: [{epoch + 1}] Avg Loss: {avg_loss:.4f} Avg Dice: {avg_dice:.4f} Avg IoU: {avg_iou:.4f}"
    )
    return avg_loss, avg_dice


def main(report_excel_path=None, lambda_sim=0.1):
    task_name = "ImageEncoder_Pretrain"
    model_name = "EviVLM"
    session_name = "Pretrain_EviVLM_" + time.strftime("%m.%d_%Hh%M")
    save_path = os.path.join(task_name, model_name, session_name)
    model_path = os.path.join(save_path, "models")
    log_path = os.path.join(save_path, session_name + ".log")
    visualize_path = os.path.join(save_path, "visualize_val")

    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    logger = logger_config(log_path)

    train_dataset_path = "D:/VLM_Medical_Imaging/dataset/Train_Folder/"
    val_dataset_path = "D:/VLM_Medical_Imaging/dataset/Val_Folder/"

    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

    train_dataset = ImageToImage2D_val(train_dataset_path, "Pretrain", train_tf, image_size=config.img_size, report_excel=report_excel_path)
    val_dataset = ImageToImage2D_val(val_dataset_path, "Pretrain", val_tf, image_size=config.img_size, report_excel=report_excel_path)

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

    device = torch.device(config.device)
    model = EviVLM(
        n_channels=config.n_channels,
        n_classes=config.n_labels,
        vision_backbone=config.vision_backbone,
        segformer_model_name=config.segformer_model_name,
        segformer_pretrained=config.segformer_pretrained,
        freeze_segformer=config.freeze_segformer,
    ).to(device)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    if config.cosineLR:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
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
            lambda_sim=lambda_sim,
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
            torch.save({
                "epoch": epoch + 1,
                "best_model": True,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(model_path, f"best_model-{model_name}.pth.tar"))

        if (epoch + 1) - best_epoch > config.early_stopping_patience:
            logger.info("\t Early stopping triggered!")
            break


if __name__ == "__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # By default look for a reports Excel in dataset root
    default_report_excel = "d:/VLM_Medical_Imaging/dataset/reports.xlsx"
    if os.path.exists(default_report_excel):
        report_path = default_report_excel
    else:
        report_path = None

    # lambda_sim can be tuned; default 0.1
    main(report_excel_path=report_path, lambda_sim=0.1)
