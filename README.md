# Color4Dial
This repository contains code and data for the paper [Dialogue Planning via Brownian Bridge Stochastic Process for Goal-directed Proactive Dialogue](https://github.com/iwangjian/Color4Dial) (to appear) accepted as Findings of ACL'2023.


## Overview
Goal-directed dialogue systems aim to proactively reach a pre-determined target through multi-turn conversations. The key to achieving this task lies in planning dialogue paths that smoothly and coherently direct conversations towards the target. In this work, we propose a **c**oherent dial**o**gue p**l**anning approach via Br**o**wnian b**r**idge (**COLOR**) stochastic process, to model the temporal dynamics of dialogue paths. We define a latent space that captures the coherence of goal-directed behavior using a Brownian bridge process, which allows us to incorporate user feedback flexibly in dialogue planning. Based on the derived latent trajectories, we generate dialogue paths explicitly using pre-trained language models. We finally employ these paths as natural language prompts to guide dialogue generation.

<p align="center">
<img src="figure/overview.png" width="100%" />
</p>


## Requirements
To be released ...

## Quickstart
To be released ...


## Citation
If you use our code or data in your work, please kindly cite our work as:
```bibtex
@inproceedings{wang2023dialogue,
  title = {Dialogue Planning via Brownian Bridge Stochastic Process for Goal-directed Proactive Dialogue},
  author = {Wang, Jian and Lin Dongding, and Li, Wenjie},
  booktitle = {Findings of Annual Meeting of the Association for Computational Linguistics (ACL)},
  year = {2023}
}
```