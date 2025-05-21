import argparse
import os
import yaml
from tqdm import tqdm

import torch
import logging
from utils import create_logging, get_dataset, get_model
from torchgeo.datasets import stack_samples, unbind_samples
from torch.utils.data import DataLoader
from dataset.garrulus import GarrulusDatasetICRA

from torchmetrics.classification import (
    MulticlassJaccardIndex,
    Dice,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.segmentation import MeanIoU


def main(cfg):
    device = torch.device("cuda", cfg["cuda"])
    de = cfg["use_dense_embeddings"]
    seed = cfg["seed"]
    peft_method = f"{cfg['peft']}_de" if de else cfg["peft"]
    peft_method = peft_method + f"_seed{seed}"
    if cfg["high_res_upsampling"]:
        peft_method = peft_method + "_highres"
    save_example = True
    # img index to save for poster visualization
    idx_img_to_save = 32

    snapshot_path, loggig = create_logging(cfg)
    model = get_model(cfg, logging, device)

    # for ICRA2025 dataset, it's loaded from already-generated samples
    # saved in tensor files
    if cfg["dataset"] == "garrulus_icra":
        val_dataset = GarrulusDatasetICRA(sampled_dataset_path=".garrulus_dataset/sampled_test_data.pt")
        valloader = DataLoader(
            val_dataset,
            batch_size=1,
            collate_fn=stack_samples,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        logging.warning("Using Garrulus ICRA 2025 dataset which is pre-generated....")
    elif cfg["dataset"] == "garrulus":
        gsd, trainloader, valloader = get_dataset(cfg)
    else:
        raise ValueError("Dataset not implemented...")

    model.eval()
    MulticlassJaccardIndex(num_classes=cfg["num_classes"]).to(device)
    dice = Dice(num_classes=cfg["num_classes"]).to(device)
    precision = MulticlassPrecision(num_classes=cfg["num_classes"], average="macro").to(
        device
    )
    recall = MulticlassRecall(num_classes=cfg["num_classes"], average="macro").to(
        device
    )

    miou = MeanIoU(
        num_classes=cfg["num_classes"],
        input_format="index",
        per_class=False,
        include_background=True,
    ).to(device)
    miou_per_class = MeanIoU(
        num_classes=cfg["num_classes"],
        input_format="index",
        per_class=True,
        include_background=True,
    ).to(device)

    for i, sampled_batch in tqdm(enumerate(valloader)):
        image = sampled_batch["image"].to(device)
        label = sampled_batch["mask"].to(device)
        outputs = model(
            batched_input=image, multimask_output=True, image_size=cfg["img_size"]
        )
        output_masks = outputs["masks"]
        pred = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)  # h,w
        dice.update(pred.squeeze(0), label.squeeze(0))
        precision.update(pred.squeeze(0), label.squeeze(0))
        recall.update(pred.squeeze(0), label.squeeze(0))
        miou.update(pred.squeeze(0), label.squeeze(0))
        miou_per_class.update(pred.squeeze(0), label.squeeze(0))

        # visualize
        if save_example and i == idx_img_to_save:
            sample = unbind_samples(sampled_batch)[0]
            sample["prediction"] = pred.cpu().detach().numpy()[0]
            fig = val_dataset.plot(sample)
            fig.savefig(f"{peft_method}_batch_sample_{i}.png", dpi=300)

        # # save result and sample
        # samples = unbind_samples(sampled_batch)
        # for sidx, sample in enumerate(samples):
        #     sample['prediction'] = pred.cpu().detach().numpy()[sidx]
        #     #print(sample['prediction'].shape)
        # segmentation_results.extend(samples)

    results = {
        "method": peft_method,
        "precision": float(precision.compute().cpu().numpy()),
        "recall": float(recall.compute().cpu().numpy()),
        "dice": float(dice.compute().cpu().numpy()),
        "mIoU": float(miou.compute().cpu().numpy()),
        "mIoU_per_class_wo_bg": float(
            miou_per_class.compute().cpu().numpy()[1:].mean()
        ),
        "IoUs": " ".join(map(str, miou_per_class.compute().cpu().numpy())),
    }
    print(results)
    logging.info("Validation finished.....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sam-vit-h-icra2025.yaml")
    parser.add_argument(
        "--peft", type=str, default="lora", help="lora, adapter_h, adapter_l"
    )
    parser.add_argument("--cuda", type=int, default=0, help="cuda device id")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--peft_ckpt", default="", help="Path to saved peft checkpoint (not SAM)"
    )

    args = parser.parse_args()

    assert args.peft_ckpt != "", "checkpoint is empty"
    assert os.path.exists(args.peft_ckpt), "Checkpoint path does not exist"

    with open(args.config, "r") as file:
        all_cfg = yaml.safe_load(file)

    cfg = all_cfg["peft"][args.peft]
    cfg["peft"] = args.peft
    cfg["peft_ckpt"] = args.peft_ckpt
    cfg["debug"] = args.debug
    cfg["cuda"] = args.cuda

    main(cfg)
