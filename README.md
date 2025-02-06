# Rethinking Large-scale Dataset Compression: Shifting Focus From Labels to Images
[[`Paper`](https://arxiv.org/abs/2502.06434) | [`BibTex`](#citation) | [`🤗Dataset`](https://huggingface.co/collections/he-yang/rethinking-large-scale-dataset-compression-67a4634cb4ed419d3a6d2720) | [`📂Logs`](https://drive.google.com/drive/folders/17tearagY46nDj1D-MfegzM9kNAOAK8SR?usp=drive_link)]

---

Official Implementation for "[Rethinking Large-scale Dataset Compression: Shifting Focus From Labels to Images](https://arxiv.org/abs/2502.06434)".

[Lingao Xiao](https://scholar.google.com/citations?user=MlNI5YYAAAAJ),&nbsp;
[Songhua Liu](https://scholar.google.com/citations?user=AnYh2rAAAAAJ),&nbsp;
[Yang He*](https://scholar.google.com/citations?user=vvnFsIIAAAAJ),&nbsp;
[Xinchao Wang](https://scholar.google.com/citations?user=w69Buq0AAAAJ)


> **Abstract**:
Dataset distillation and dataset pruning are two prominent techniques for compressing datasets to improve computational and storage efficiency. 
Despite their overlapping objectives, these approaches are rarely compared directly. 
Even within each field, the evaluation protocols are inconsistent across various methods, which complicates fair comparisons and hinders reproducibility. 
Considering these limitations, we introduce in this paper a benchmark that equitably evaluates methodologies across both distillation and pruning literatures. 
Notably, our benchmark reveals that in the mainstream dataset distillation setting for large-scale datasets, which heavily rely on soft labels from pre-trained models, even randomly selected subsets can achieve surprisingly competitive performance.
This finding suggests that an overemphasis on soft labels may be diverting attention from the intrinsic value of the image data, while also imposing additional burdens in terms of generation, storage, and application. 
To address these issues, we propose a new framework for dataset compression, termed *Prune, Combine, and Augment (PCA)*, which focuses on leveraging image data exclusively, relies solely on hard labels for evaluation, and achieves state-of-the-art performance in this setup. 
By shifting the emphasis back to the images, our benchmark and PCA framework pave the way for more balanced and accessible techniques in dataset compression research. 

## TODOs
- [x] release large-scale benchmark
- [x] release SOTA datasets
- [ ] release PCA framework
- [ ] release PCA datasets

*Note: for soft label benchmark, we use [fast evaluation code without relabeling](https://github.com/VILA-Lab/SRe2L/tree/main/SRe2L/validate#alternative-validation).

## Datasets ([🤗Hugging Face](https://huggingface.co/collections/he-yang/rethinking-large-scale-dataset-compression-67a4634cb4ed419d3a6d2720))

SOTA datasets used in our experiments are available at [🤗Hugging Face](https://huggingface.co/collections/he-yang/rethinking-large-scale-dataset-compression-67a4634cb4ed419d3a6d2720).
We have preprocessed all images into fixed 224x224 resolutioins and creates the datasets for a fair storage comparison.

| Type    |         `DD`          |       `DP`       |
| ------- | :------------------: | :-------------: |
| Explain | Dataset Distillation | Dataset Pruning |

| Method                                         | Type | Venue      | Dataset Key                                            | Avaiable IPCs          |
| ---------------------------------------------- | ---- | ---------- | ------------------------------------------------------ | ---------------------- |
| random                                         | -    | -          | he-yang/2025-rethinkdc-imagenet-random-ipc-`[IPC]`     | `[1,10,20,50,100,200]` |
| [SRe2L](https://arxiv.org/abs/2306.13092)      | `DD`  | NeurIPS'23 | he-yang/2025-rethinkdc-imagenet-sre2l-ipc-`[IPC]`      | `[10,50,100]`          |
| [CDA](https://arxiv.org/abs/2311.18838)        | `DD`  | TMLR'24    | he-yang/2025-rethinkdc-imagenet-cda-ipc-`[IPC]`        | `[10,50,100]`          |
| [G-VBSM](https://arxiv.org/abs/2311.17950)     | `DD`  | CVPR'24    | he-yang/2025-rethinkdc-imagenet-gvbsm-ipc-`[IPC]`      | `[10,50,100]`          |
| [LPLD](https://arxiv.org/abs/2410.15919)       | `DD`  | NeurIPS'24 | he-yang/2025-rethinkdc-imagenet-lpld-ipc-`[IPC]`       | `[10,50,100]`          |
| [RDED](https://arxiv.org/abs/2312.03526)       | `DD`  | CVPR'24    | he-yang/2025-rethinkdc-imagenet-rded-ipc-`[IPC]`       | `[10,50,100]`          |
| [DWA](https://arxiv.org/abs/2409.17612)        | `DD`  | NeurIPS'24 | he-yang/2025-rethinkdc-imagenet-dwa-ipc-`[IPC]`        | `[10,50,100]`          |
| [Forgetting](https://arxiv.org/abs/1812.05159) | `DP`  | ICLR'19    | he-yang/2025-rethinkdc-imagenet-forgetting-ipc-`[IPC]` | `[10,50,100]`          |
| [EL2N](https://arxiv.org/abs/2107.07075)       | `DP`  | NeurIPS'21 | he-yang/2025-rethinkdc-imagenet-el2n-ipc-`[IPC]`       | `[10,50,100]`          |
| [AUM](https://arxiv.org/abs/2001.10528)        | `DP`  | NeurIPS'20 | he-yang/2025-rethinkdc-imagenet-aum-ipc-`[IPC]`        | `[10,50,100]`          |
| [CCS](https://arxiv.org/abs/2210.15809)        | `DP`  | ICLR'23    | he-yang/2025-rethinkdc-imagenet-ccs-ipc-`[IPC]`        | `[10,50,100]`          |

## Installation

1\. Install from pip (tested on python=3.12)

```sh
pip install rethinkdc
```


<details>
<summary>2. Or install from source</summary>

**Step 1**: Clone Repo,
```sh
git clone https://github.com/ArmandXiao/Rethinking-Dataset-Compression.git
cd Rethinking-Dataset-Compression
```

**Step 2**: Create Environment,
```sh
conda env create -f environment.yml
conda activate rethinkdc
```

**Step 3**: Install Benchmark,
```sh
make build
make install
```
</details>

## Usage
1\. Prepare ImageNet Validation Folder:
```sh
# download and prepare ImageNet Val (skip if you have)
wget -qO- https://github.com/ArmandXiao/Rethinking-Dataset-Compression/script/download_val.sh | bash

# set environment (IMPORTANT!)
export IMAGENET_VAL_DIR="Your ImageNet Val Path"
```

2\. Hyper-parameter for "rethinkdc"
```sh
rethinkdc --help
```

<details>
<summary>📘 Manual</summary>

```
Rethinking Large-scale Dataset Compression
usage: rethinkdc [-h] [--soft | --hard | --yaml YAML] [--batch-size BATCH_SIZE] [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] [-j WORKERS]
                 [--val-dir VAL_DIR] [--output-dir OUTPUT_DIR] [--hf-cache-dir HF_CACHE_DIR] [--mode MODE] [--cos] [--adamw-lr ADAMW_LR]
                 [--adamw-weight-decay ADAMW_WEIGHT_DECAY] [--sgd-setting] [--hard-label] [--start-epoch START_EPOCH] [--epochs EPOCHS] [--model MODEL]
                 [--teacher-model TEACHER_MODEL] [-T TEMPERATURE] [--mix-type MIX_TYPE] [--mixup MIXUP] [--cutmix CUTMIX] [--ipc IPC] [--wandb-project WANDB_PROJECT]
                 [--wandb-api-key WANDB_API_KEY]
                 

Example Usage:
        rethinkdc he-yang/2025-rethinkdc-imagenet-random-ipc-10 --soft --ipc 10 --output-dir ./random_ipc10_soft 

                                                      options                                                       
-h, --help          ┃ show this help message and exit                            ┃ str    ┃ ==SUPPRESS==            
                                               Configuration Options                                                
--soft              ┃ Use standard_soft_config.yaml (Example: rethinkdc PATH     ┃ str    ┃ False                   
                    ┃ --soft)                                                    ┃        ┃                         
--hard              ┃ Use standard_hard_config.yaml (Example: rethinkdc PATH     ┃ str    ┃ False                   
                    ┃ --hard)                                                    ┃        ┃                         
--yaml              ┃ Custom config file (Exmpale: rethinkdc                     ┃ str    ┃                         
                    ┃ YOUR_PATH_TO_CONFIG.yaml)                                  ┃        ┃                         
                                                    Data Options                                                    
train_dir           ┃ path to training dataset or huggingface dataset key        ┃ str    ┃                         
--batch-size        ┃ batch size                                                 ┃ int    ┃ 1024                    
--gradient-accumul… ┃ gradient accumulation steps for small gpu memory           ┃ int    ┃ 1                       
-j, --workers       ┃ number of data loading workers                             ┃ int    ┃ 16                      
--val-dir           ┃ path to validation dataset                                 ┃ str    ┃ /path/to/imagenet/val   
--output-dir        ┃ path to output dir                                         ┃ str    ┃ ./save/1024             
--hf-cache-dir      ┃ cache dir for huggingface dataset                          ┃ str    ┃ ./hf_cache              
--mode              ┃ mode for training                                          ┃ str    ┃ fkd_save                
                                                  Training Options                                                  
--cos               ┃ cosine lr scheduler                                        ┃ str    ┃ False                   
--adamw-lr          ┃ adamw learning rate                                        ┃ float  ┃ 0.001                   
--adamw-weight-dec… ┃ adamw weight decay                                         ┃ float  ┃ 0.01                    
--sgd-setting       ┃ using sgd evaluation settting (lr=0.1, scheduler=cos)      ┃ str    ┃ False                   
--hard-label        ┃ use hard label                                             ┃ str    ┃ False                   
--start-epoch       ┃ start epoch                                                ┃ int    ┃ 0                       
--epochs            ┃ total epoch                                                ┃ int    ┃ 300                     
                                                   Model Options                                                    
--model             ┃ student model name                                         ┃ str    ┃ resnet18                
--teacher-model     ┃ teacher model name                                         ┃ str    ┃                         
-T, --temperature   ┃ temperature for distillation loss                          ┃ float  ┃ 3.0                     
                                                Mixup/CutMix Options                                                
--mix-type          ┃ choices in {mixup, cutmix, None}                           ┃ str    ┃                         
--mixup             ┃ mixup alpha, mixup enabled if > 0. (default: 0.8)          ┃ float  ┃ 0.8                     
--cutmix            ┃ cutmix alpha, cutmix enabled if > 0. (default: 1.0)        ┃ float  ┃ 1.0                     
--ipc               ┃ number of images per class                                 ┃ int    ┃ 50                      
                                                   Wandb Options                                                    
--wandb-project     ┃ wandb project name                                         ┃ str    ┃ Temperature             
--wandb-api-key     ┃ wandb api key                                              ┃ str    ┃                         

For more information, please visit the project repository: https://github.com/ArmandXiao/Rethinking-Dataset-Compression
```

</details>


3\. Example Usage (more examples can be found in folder [script](script)):
```sh
rethinkdc [YOUR_PATH_TO_DATASET] [*ARGS]

# example (test random subset)
rethinkdc he-yang/2025-rethinkdc-imagenet-random-ipc-10 --soft --ipc 10 --output-dir ./random_ipc10_soft
```

## Main Table Result ([📂Google Drive](https://drive.google.com/drive/folders/17tearagY46nDj1D-MfegzM9kNAOAK8SR?usp=drive_link))

Logs for main tables are results provided in [google drive](https://drive.google.com/drive/folders/17tearagY46nDj1D-MfegzM9kNAOAK8SR?usp=drive_link) for reference.

| Table                                                                                                          | Explanation                                                     |
| -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [Table 3](https://drive.google.com/drive/folders/1ZlqoLPmMV235F4G3NyCMM1af_VTHPjA7?usp=drive_link)             | Random baselines in soft label setting with standard evaluation |
| [Table 4 & Table 18](https://drive.google.com/drive/folders/1Zs25THv54VNYcJ72KeyABjMZYL7hFbxT?usp=drive_link)  | SOTA methods in soft label setting with std                     |
| [Table 5 & Table 19](https://drive.google.com/drive/folders/1T1xsCWA9ahhICAeTBiVLlYX-G0-bw6gy?usp=drive_link)  | SOTA methods in hard label setting with std                     |
| [Table 6](https://drive.google.com/drive/folders/1rtEnoO8TUteg5E5wS1vLazuix-B0r3ph?usp=drive_link)             | SOTA Pruning Rules                                              |
| [Table 7](https://drive.google.com/drive/folders/13OBiPnBA8y2iCu-9C63d0iR6jdEKKrTB?usp=drive_link)             | Ablation Study of PCA                                           |
| [Table 8](https://drive.google.com/drive/folders/1-QTCzBEgQDw_RtyYXSy0zYOZcvAG7BCo?usp=drive_link)             | Cross-architecture Performance of PCA                           |
| [Table 12 & Table 22](https://drive.google.com/drive/folders/1YKFPAtmnoFAQipLd2YNlSsoNColf46aU?usp=drive_link) | Regularization-based Data Augmentation                          |
| [Table 20](https://drive.google.com/drive/folders/1O4dt67os89kHvVROcjYgNSLtMHQU4V6K?usp=drive_link)            | Pure Noise as Input                                             |
| [Table 24](https://drive.google.com/drive/folders/1PW2Pf8o7f_3ZvCIvyU6-Rd-9mFMdxetn?usp=drive_link)            | PCA using Different Pruning Methods                             |


## Related Repos
- [https://github.com/VILA-Lab/SRe2L](https://github.com/VILA-Lab/SRe2L)
- [https://github.com/he-y/soft-label-pruning-for-dataset-distillation](https://github.com/he-y/soft-label-pruning-for-dataset-distillation)
- [https://github.com/haizhongzheng/Coverage-centric-coreset-selection](https://github.com/haizhongzheng/Coverage-centric-coreset-selection)
- [https://github.com/NUS-HPC-AI-Lab/DD-Ranking](https://github.com/NUS-HPC-AI-Lab/DD-Ranking)


## Citation
```
@article{xiao2025rethinkdc,
  title={Rethinking Large-scale Dataset Compression: Shifting Focus From Labels to Images},
  author={Xiao, Lingao and Liu, Songhua and He, Yang and Wang, Xinchao},
  journal={arXiv preprint arXiv:2502.06434},
  year={2025}
}
```