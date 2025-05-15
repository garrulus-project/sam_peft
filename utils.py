import os
import sys
import numpy as np
import logging

import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from torchmetrics.classification import (
    Dice,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import MeanIoU

import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    # TODO: THIS ONE HOT ENCODER DOES NOT MAP PROPERLY
    # IT MAPS 5 CLASSES TO 6: eg. [4,255,55] to [4, 6, 512, 512] instead of [4, 5, 512, 512]
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        # target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), (
            "inputs {} & target {} shape do not match".format(
                inputs.size(), target.size()
            )
        )
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def evaluate(model, valloader, num_classes, device, cfg):
    model.eval()
    # jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    # device = 'cpu'
    dice = Dice(num_classes=num_classes).to(device)
    precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    miou = MeanIoU(
        num_classes=num_classes,
        input_format="index",
        per_class=False,
        include_background=True,
    ).to(device)
    miou_class = MeanIoU(
        num_classes=num_classes,
        input_format="index",
        per_class=True,
        include_background=False,
    ).to(device)

    # Reset metrics before evaluation
    dice.reset()
    precision.reset()
    recall.reset()
    miou.reset()
    miou_class.reset()

    # always use no_grad when evaluating, so that it reduces the mem use
    with torch.no_grad():
        for i, sampled_batch in enumerate(valloader):
            image = sampled_batch["image"].to(device)
            label = sampled_batch["mask"].to(device)
            outputs = model(
                batched_input=image, multimask_output=True, image_size=cfg["img_size"]
            )  # inputs 1,c,h,w
            output_masks = outputs["masks"]  # 1,2,h,w
            pred = torch.argmax(
                torch.softmax(output_masks, dim=1), dim=1
            ).squeeze()  # h,w

            # jaccard.update(pred, label.squeeze(0))

            label = label.detach().to(device)
            pred = pred.detach().to(device)
            dice.update(pred, label.squeeze(0))
            precision.update(pred, label.squeeze(0))
            recall.update(pred, label.squeeze(0))

            miou.update(pred, label.squeeze(0))
            miou_class.update(pred, label.squeeze(0))

    results = {
        "precision": float(precision.compute().cpu().numpy()),
        "recall": float(recall.compute().cpu().numpy()),
        "dice": float(dice.compute().cpu().numpy()),
        "mIoU": float(miou.compute().cpu().numpy()),
        "class_mIoU": float(miou_class.compute().cpu().numpy().mean()),
    }
    return results


def evaluate_dist(model, valloader, num_classes, device, cfg):
    """
    Evaluation on distributed training
    """
    model.eval()
    # jaccard = MulticlassJaccardIndex(num_classes=num_classes).to(device)
    metric_device = device
    dice = Dice(num_classes=num_classes).to(metric_device)
    precision = MulticlassPrecision(num_classes=num_classes, average="macro").to(
        metric_device
    )
    recall = MulticlassRecall(num_classes=num_classes, average="macro").to(
        metric_device
    )
    miou = MeanIoU(
        num_classes=num_classes,
        input_format="index",
        per_class=False,
        include_background=True,
    ).to(metric_device)
    miou_class = MeanIoU(
        num_classes=num_classes,
        input_format="index",
        per_class=True,
        include_background=False,
    ).to(metric_device)

    # Reset metrics before evaluation
    dice.reset()
    precision.reset()
    recall.reset()
    miou.reset()
    miou_class.reset()

    with (
        torch.no_grad()
    ):  # always use no_grad when evaluating, so that it reduces the mem use
        for i, sampled_batch in enumerate(valloader):
            image = sampled_batch["image"].to(device)
            label = sampled_batch["mask"].to(device)
            outputs = model(
                batched_input=image, multimask_output=True, image_size=cfg["img_size"]
            )  # inputs 1,c,h,w
            output_masks = outputs["masks"]  # 1,2,h,w
            pred = torch.argmax(
                torch.softmax(output_masks, dim=1), dim=1
            ).squeeze()  # h,w

            # jaccard.update(pred, label.squeeze(0))

            label = label.detach().to(metric_device)
            pred = pred.detach().to(metric_device)
            dice.update(pred, label.squeeze(0))
            precision.update(pred, label.squeeze(0))
            recall.update(pred, label.squeeze(0))

            miou.update(pred, label.squeeze(0))
            miou_class.update(pred, label.squeeze(0))

    # Now compute the scalar value for each metric
    dice_value = dice.compute()
    precision_value = precision.compute()
    recall_value = recall.compute()
    miou_value = miou.compute()
    miou_class_value = miou_class.compute()

    # Convert the computed values to tensors for all-reduce
    dice_tensor = torch.tensor(dice_value).to(device)
    precision_tensor = torch.tensor(precision_value).to(device)
    recall_tensor = torch.tensor(recall_value).to(device)
    miou_tensor = torch.tensor(miou_value).to(device)
    miou_class_tensor = torch.tensor(miou_class_value).to(device)

    # Reduce metrics across all processes
    def reduce_metric(tensor):
        dist.all_reduce(
            tensor, op=dist.ReduceOp.SUM
        )  # Sum the values across all processes
        tensor /= dist.get_world_size()  # Average the values by the number of processes

    # Aggregate metrics across all ranks
    reduce_metric(dice_tensor)
    reduce_metric(precision_tensor)
    reduce_metric(recall_tensor)
    reduce_metric(miou_tensor)
    reduce_metric(miou_class_tensor)

    # Only rank 0 will return the final results
    if dist.get_rank() == 0:
        results = {
            "precision": float(precision_tensor.cpu().numpy()),
            "recall": float(recall_tensor.cpu().numpy()),
            "dice": float(dice_tensor.cpu().numpy()),
            "mIoU": float(miou_tensor.cpu().numpy()),
            "class_mIoU": float(miou_class_tensor.cpu().numpy().mean()),
        }
        return results
    else:
        return None


def calc_loss(outputs, ground_truth, ce_loss, dice_loss, dice_weight: float = 0.8):
    masks = outputs["masks"]
    loss_ce = ce_loss(masks, ground_truth.long())
    loss_dice = dice_loss(masks, ground_truth, softmax=True)
    # generalize dice loss
    # loss_dice = dice_loss(masks, high_res_label_batch)
    loss = ((1 - dice_weight) * loss_ce) + (dice_weight * loss_dice)
    return loss, loss_ce, loss_dice


def get_sam_model_reg_key(sam_ckpt):
    """
    Get same model registry key given sam checkpoint file
    """
    for reg_key in sam_model_registry.keys():
        if reg_key in sam_ckpt:
            print("Sam registry key: ", reg_key)
            return reg_key


def worker_init_fn(worker_id, seed):
    np.random.seed(seed + worker_id)


def get_step_scheduler_with_warmup(
    optimizer, warmup_steps, initial_lr, decay_steps, decay_factor
):
    """
    Returns a learning rate scheduler with a warmup phase followed by step-wise decay.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        warmup_steps (int): The number of steps for the warmup phase.
        initial_lr (float): The initial learning rate after warmup.
        decay_steps (int): The number of steps between each decay.
        decay_factor (float): The factor by which to decay the learning rate.

    Returns:
        scheduler (torch.optim.lr_scheduler.LambdaLR): A learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Step-wise decay
            num_decays = (current_step - warmup_steps) // decay_steps
            return decay_factor**num_decays

    return LambdaLR(optimizer, lr_lambda)


class LinearWarmupLR(LambdaLR):
    """Linearly increases learning rate from 0 to the provided value over `warmup_steps` training steps.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`): The optimizer to apply the schedule to.
        warmup_steps (:obj:`int`): The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1): The index of the last epoch when resuming training.
    """

    def __init__(
        self, optimizer: optim.Optimizer, warmup_steps: int, last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


def create_logging(cfg):
    log_path = os.path.join(cfg["output"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    sam_registry_key = get_sam_model_reg_key(cfg["sam_ckpt"])
    print("SAM model registry key: ", sam_registry_key)

    snapshot_path = f"_sam_{sam_registry_key}"
    snapshot_path = snapshot_path + f"_epoch{cfg['max_epochs']}"
    snapshot_path = snapshot_path + f"_bs{cfg['batch_size']}"
    snapshot_path = snapshot_path + f"_seed{cfg['seed']}"
    snapshot_path = snapshot_path + f"_{cfg['peft']}"
    if cfg["use_dense_embeddings"]:
        snapshot_path = snapshot_path + "_de"

    if "adapter" in cfg["peft"]:
        snapshot_path = snapshot_path + "_dim" + str(cfg["middle_dim"])
        snapshot_path = snapshot_path + "_sf" + str(cfg["scaling_factor"])
    elif "lora" in cfg["peft"]:
        snapshot_path = snapshot_path + "_r" + str(cfg["rank"])

    if cfg["debug"]:
        snapshot_path = "debug_" + snapshot_path

    snapshot_path = os.path.join(log_path, snapshot_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    config_file = os.path.join(snapshot_path, "config.txt")
    config_items = []
    for key in cfg:
        config_items.append(f"{key}: {cfg[key]}\n")

    with open(config_file, "w") as f:
        f.writelines(config_items)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg))

    return snapshot_path, logging
