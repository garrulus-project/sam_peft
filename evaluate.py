import argparse
import os
import yaml

import torch
import logging
from utils import evaluate
from utils import create_logging
from utils import get_model, get_dataset


def main(cfg):
    device = torch.device("cuda", cfg["cuda"])

    snapshot_path, loggig = create_logging(cfg)
    model = get_model(cfg, logging, device)

    gsd, trainloader, valloader = get_dataset(cfg)

    cfg["num_classes"]
    model.train()

    results = evaluate(model, valloader, cfg["num_classes"], device, cfg)

    logging.info(
        f"Precision: {results['precision']} Recall: {results['recall']} Dice: {results['dice']}  mIOU : {results['mIoU']}"
    )
    logging.info("Validation in epoch Finished!")

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
    