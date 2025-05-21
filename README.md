# Parameter-Efficient Fine-Tuning of Vision Foundation Model for Forest Floor Segmentation from UAV Imagery

This is the official implementation of our paper **"Parameter-Efficient Fine-Tuning of Vision Foundation Model for Forest Floor Segmentation from UAV Imagery"**, [ICRA 2025 Workshop on the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots](https://ag-tech-icra2025.com/).

**Abstract**

Unmanned Aerial Vehicles (UAVs) are increasingly used for reforestation and forest monitoring, 
including seed dispersal in hard-to-reach terrains. However, a detailed understanding of 
the forest floor remains a challenge due to high natural variability, quickly changing 
environmental parameters, and ambiguous annotations due to unclear definitions. 
To address this issue, we adapt the Segment Anything Model (SAM), a vision foundation model 
with strong generalization capabilities, to segment forest floor objects such as tree stumps, 
vegetation, and woody debris. To this end, we employ parameter-efficient fine-tuning (PEFT) 
to fine-tune a small subset of additional model parameters while keeping the original 
weights fixed. We adjust SAM's mask decoder to generate masks corresponding 
to our dataset categories, allowing for automatic segmentation without manual prompting. 
Our results show that the adapter-based PEFT method achieves 
the highest mean intersection over union (mIoU), while Low-rank Adaptation (LoRA), 
with fewer parameters, offers a lightweight alternative for resource-constrained UAV platforms. 

## Requirements

### Environment

* Create conda env and activate
  ```
  conda create -n myenv python=3.11
  conda activate myenv
  ```
* Install dependencies
  ```
  pip install -r requirements.txt
  ```

### Dataset
* Download dataset [here](https://zenodo.org/records/15480886)
* Create a directory `garrulus_dataset` and move both train and test datasets there
  ```
  mkdir garrulus_dataset
  ```

### Model checkpoints
* Download pretrained sam-vit-h (`sam_vit_h_4b8939.pth`) model [here](https://github.com/facebookresearch/segment-anything)
* Create a directory `checkpoints/sam` and move the model there
  ```
  mkdir -p checkpoints/sam
  mv sam_vit_h_4b8939.pth checkpoints/sam
  ```

## Training

Each experiment was conducted on a single NVIDIA RTX A5000 24GB

* Train PEFT methods and SAM mask decoder
  ```
  # train adapter_h
  python train.py --config config/sam-vit-h-icra2025.yaml --peft adapter_h --seed=42 --cuda=0
  
  # train adapter_l
  python train.py --config config/sam-vit-h-icra2025.yaml --peft adapter_l --seed=42 --cuda=0

  # train lora
  python train.py --config config/sam-vit-h-icra2025.yaml --peft lora --seed=42 --cuda=0

  # train sam_decoder
  python train.py --config config/sam-vit-h-icra2025.yaml --peft sam_decoder --seed=42 --cuda=0
  ```

## Evaluation
  ```
  python evaluate.py --config config/sam-vit-h-icra2025.yaml --peft adapter_h --cuda=0 \
  --peft_ckpt /path/to/peft_ckpt/ 
  ```

## Citation
```bibtex
@misc{wasil2025peftsam,
  title     = {{Parameter-Efficient Fine-Tuning of Vision Foundation Model for Forest Floor Segmentation from UAV Imagery}},
  author    = {Mohammad Wasil and Ahmad Drak and Brennan Penfold and Ludovico Scarton and Maximilian Johenneken and Alexander Asteroth and Sebastian Houben},
  year      = {2025},
  eprint    = {2505.08932},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url       = {https://arxiv.org/abs/2505.08932},
  note      = {Accepted to the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots, IEEE ICRA Workshop 2025}
}

```
