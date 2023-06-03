# Color4Dial
This repository contains code and data for the paper [Dialogue Planning via Brownian Bridge Stochastic Process for Goal-directed Proactive Dialogue](https://arxiv.org/abs/2305.05290) accepted as Findings of ACL-2023.


## Overview
Goal-directed dialogue systems aim to proactively reach a pre-determined target through multi-turn conversations. The key to achieving this task lies in planning dialogue paths that smoothly and coherently direct conversations towards the target. In this work, we propose a **c**oherent dial**o**gue p**l**anning approach via Br**o**wnian b**r**idge (**COLOR**) stochastic process, to model the temporal dynamics of dialogue paths. We define a latent space that captures the coherence of goal-directed behavior using a Brownian bridge process, which allows us to incorporate user feedback flexibly in dialogue planning. Based on the derived latent trajectories, we generate dialogue paths explicitly using pre-trained language models. We finally employ these paths as natural language prompts to guide dialogue generation.

<p align="center">
<img src="figure/overview.png" width="100%" />
</p>


## Requirements
Suppose [Anaconda](https://www.anaconda.com/) is used to manage the environment. The required packages are listed in `requirements.txt`. You can install them by running:
```bash
conda create -n color4dial python=3.10
conda activate color4dial
pip install -r requirements.txt
```

## Datasets
We upload the datasets used in our experiments to the OneDrive cloud storage. Please download [DuRecdial 2.0](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21037774r_connect_polyu_hk/EUX6GBdtYJRNuZ4HY-Y9Q30BtLitoxiOhZY3cCI9Y_b9dQ?e=wegoOQ) and [TGConv](https://connectpolyu-my.sharepoint.com/:u:/g/personal/21037774r_connect_polyu_hk/EZ2XZ49qdwhGoMKsXx7GWxgBaIQKGNPphAi7NUhXV6hcyw?e=wjrB7l) datasets and put them in the `data` folder.
```bash
cd data
unzip DuRecdial2.zip & rm DuRecdial2.zip
unzip TGConv.zip & rm TGConv.zip
```

## Quickstart
Take the DuRecdial 2.0 dataset as an example, our experiments are divided into three stages.

### Stage 1: Brownian Bridge Mapping
In this stage, we learn a mapping in the Brownian bridge latent space that captures coherent temporal dynamics for planning dialogue paths.
```bash
bash scripts/durecdial_planning_train_bridge.sh
```
For more details of parameter settings, please refer to `main_planning.py`.

### Stage 2: Planning Dialogue Paths
Based on the learned Brownian bridge mapping, we train a planner model and use it to plan dialogue paths.
```bash
# model training
bash scripts/durecdial_planning_train_planner.sh

# model inference
bash scripts/durecdial_planning_infer_planner.sh
```
For more details of parameter settings, please refer to `main_planning.py`.

### Stage 3: Generating Dialogue Utterances
Finally, we employ the planned dialogue paths as natural language prompts to guide dialogue generation.
```bash
# model training
bash scripts/durecdial_dialog_train.sh

# model inference
bash scripts/durecdial_dialog_test.sh
```
For more details of parameter settings, please refer to `main_dialog.py`.

## Evaluation
To evaluate the performance dialogue planning, please run:
```python
python eval/eval_planning.py --dataset <dataset_name> \
  --eval_file <path_to_eval> \
  --gold_file <path_to_gold_data>
```
To evaluate the performance of dialogue generation, please run:
```python
# for DuRecdial 2.0 dataset
python eval/eval_dialog_durecdial.py --eval_file <path_to_eval> \
--gold_file <path_to_gold_data>

# for TGConv dataset
python eval/eval_dialog_tgconv_selfplay.py --eval_file <path_to_eval>
```

## Acknowledgement
Our code is based on parts of the implementations of [Huggingface Transformers](https://github.com/huggingface/transformers) and [Language Modeling via Stochastic Processes](https://github.com/rosewang2008/language_modeling_via_stochastic_processes). We thank the authors for their excellent work.


## Citation
If you use our data or code in your work, please kindly cite our work as:
```bibtex
@inproceedings{wang2023dialogue,
  title = {Dialogue Planning via Brownian Bridge Stochastic Process for Goal-directed Proactive Dialogue},
  author = {Wang, Jian and Lin Dongding, and Li, Wenjie},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  year = {2023}
}
```