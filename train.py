import argparse
import os
import random
import numpy as np
import yaml

import torch
import torch.backends.cudnn as cudnn
from models.segment_anything import sam_model_registry

import logging

import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm
from utils import DiceLoss
from utils import evaluate

from models.peft import (
    adapter_h,
    adapter_l,
    lora,
    sam_decoder,
)

from utils import (
    calc_loss,
    get_sam_model_reg_key,
    create_logging,
    get_model,
    get_dataset
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.garrulus import get_train_val
from utils import LinearWarmupLR

# multimasks output always True in this exp
multimask_output = True

def train(cfg):
    device = torch.device("cuda", cfg["cuda"])

    snapshot_path, loggig = create_logging(cfg)
    model = get_model(cfg, logging, device)

    gsd, trainloader, valloader = get_dataset(cfg)

    base_lr = cfg["base_lr"]
    num_classes = cfg["num_classes"]
    cfg["batch_size"] * cfg["n_gpu"]

    if cfg["n_gpu"] > 1:
        # easier but less efficient, see train_distributed.py (WIP) for more scalability training
        model = nn.DataParallel(model)

    # define loss
    ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(num_classes + 1)
    dice_loss = DiceLoss(num_classes)

    # filter optimizer to performs updates on params with requires_grad only
    # ToDO: AdamW does not work and loss does not decrease
    if cfg["AdamW"]:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            momentum=0.9,
            weight_decay=0.0001,
        )

    # define scheduler
    max_iterations = cfg["max_epochs"] * len(trainloader)

    min_lr_ratio = 0.01
    min_lr = base_lr * min_lr_ratio
    warmup_ratio = 0.1  # 10% warmup
    warmup_iterations = int(warmup_ratio * max_iterations)
    warmup_scheduler = LinearWarmupLR(optimizer, warmup_iterations)

    # coside scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max_iterations - warmup_iterations, eta_min=min_lr
    )

    def combined_scheduler(step):
        if step < warmup_iterations:
            warmup_scheduler.step()
            return warmup_scheduler.get_last_lr()[0]
        else:
            cosine_scheduler.step()
            return cosine_scheduler.get_last_lr()[0]

    # Automatic mixed precision,e.g. perform operations from float32 to float16 automatically
    # Using amp may cause unstable gradients during training
    if cfg["use_amp"]:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg["use_amp"])

    iter_num = 0
    logging.info(
        "{} iterations per epoch. {} max iterations ".format(
            len(trainloader), max_iterations
        )
    )

    track_best_model = True
    total_iter = iter_num

    best_miou = -1.0

    # start training
    for epoch_num in range(cfg["max_epochs"]):
        model.train()

        max_iter_per_epoch = len(trainloader)
        progress_bar = tqdm(
            enumerate(trainloader),
            total=max_iter_per_epoch,
            desc=f"Epoch {epoch_num}: [{iter_num}/{max_iter_per_epoch}] loss: {0:.5f} loss_ce: {0:.5f} loss_dice: {0:0.5f} lr: {base_lr:0.9f}",
        )

        for i_batch, sampled_batch in progress_bar:
            step = epoch_num * max_iter_per_epoch + i_batch

            optimizer.zero_grad()

            image_batch = sampled_batch["image"].to(device)  # [b, c, h, w]
            label_batch = sampled_batch["mask"].to(device)  # [b, h, w]

            if cfg["use_amp"]:
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=cfg["use_amp"]
                ):
                    outputs = model(image_batch, multimask_output, cfg["img_size"])
                    loss, loss_ce, loss_dice = calc_loss(
                        outputs, label_batch, ce_loss, dice_loss, cfg["dice_param"]
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(image_batch, multimask_output, cfg["img_size"])
                loss, loss_ce, loss_dice = calc_loss(
                    outputs, label_batch, ce_loss, dice_loss, cfg["dice_param"]
                )
                loss.backward()
                optimizer.step()

            # update scheduler and lr
            last_lr = combined_scheduler(step)

            iter_num = iter_num + 1
            total_iter += 1

            # update pbar
            progress_bar.set_description(
                f"Epoch {epoch_num}: [{iter_num}/{max_iter_per_epoch}] loss: {loss:.5f} loss_ce: {loss_ce:.5f} loss_dice: {loss_dice:0.5f} lr: {last_lr:0.9f}"
            )

            if iter_num % 100 == 0:
                logging.info(
                    "iteration %d: loss: %f, loss_ce: %f, loss_dice: %f  lr: %f"
                    % (
                        total_iter,
                        loss.item(),
                        loss_ce.item(),
                        loss_dice.item(),
                        last_lr,
                    )
                )

            if cfg["debug"] and iter_num > 5:
                break

        iter_num = 0
        val_interval = cfg["save_interval"]
        if epoch_num % val_interval == 0 or epoch_num >= cfg["max_epochs"] - 1:
            results = evaluate(model, valloader, cfg["num_classes"], device, cfg)
            logging.info(
                f"Precision: {results['precision']} Recall: {results['recall']} Dice: {results['dice']}  mIOU : {results['mIoU']}"
            )
            logging.info("Validation in epoch %d Finished!" % epoch_num)

            save_model_path = os.path.join(
                snapshot_path, "epoch_" + str(epoch_num) + ".pth"
            )
            model.save_peft_parameters(save_model_path)

            # init best model
            if epoch_num == 0:
                save_best_model_path = os.path.join(snapshot_path, "best.pth")
                model.save_peft_parameters(save_best_model_path)
                best_miou = results["mIoU"]

            if epoch_num >= cfg["max_epochs"] - 1:
                model.save_peft_parameters(os.path.join(snapshot_path, "last.pth"))

            logging.info("save model to {}".format(save_model_path))

            # keep track of best model every eval_interval
            if track_best_model and epoch_num > 0:
                if best_miou < results["mIoU"]:
                    best_miou = results["mIoU"]
                    model.save_peft_parameters(os.path.join(snapshot_path, "best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sam-vit-h.yaml")
    parser.add_argument(
        "--peft", type=str, default="lora", help="lora, adapter_h, adapter_l"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed (default uses the config seed)"
    )
    parser.add_argument("--cuda", type=int, default=0, help="cuda device id")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(args.config, "r") as file:
        all_cfg = yaml.safe_load(file)

    cfg = all_cfg["peft"][args.peft]
    cfg["peft"] = args.peft

    if args.seed > 0:
        cfg["seed"] = args.seed

    cfg["debug"] = args.debug
    cfg["cuda"] = args.cuda

    print("debuggggg: ", args.debug)

    if cfg["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg["deterministic"]:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])

    train(cfg)

    print("Finished training....")
