# Aligning Distributionally Robust Optimization with Practical Deep Learning Needs
:scroll: [arXiv](https://arxiv.org/pdf/2508.16734)
&nbsp; :computer: [Usage](#using-also-in-practice)

*TL;DR: ALSO is adaptive distributionally robust optimizer, which is compatible with DL pipelines.*

This repository contains the official implementation for the paper ["Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"](https://arxiv.org/pdf/2508.16734)

The repository provides:
- The official implementation of the ALSO method.
- The code for experiments used in the paper.

## Abstract:

Deep learning (DL) models often struggle with real-world data heterogeneity, such as class imbalance or varied data sources, as standard training methods treat all samples equally. Distributionally Robust Optimization (DRO) offers a principled approach by optimizing for a worst-case data distribution. However, a significant gap exists between DRO and current DL practices. DRO methods often lack adaptive parameter updates (like Adam), struggle with the non-convexity of neural networks, and are difficult to integrate with group-based weighting in standard mini-batch training pipelines. This paper aims to bridge this gap by introducing ALSO – Adaptive Loss Scaling Optimizer – a novel optimizer that integrates an adaptive, Adam-like update for the model parameters with an efficient, principled mechanism for learning worst-case data weights. Crucially, it supports stochastic updates for both model parameters and data weights, making it fully compatible with group-based weighting and standard Deep Learning training pipelines. We prove the convergence of our proposed algorithm for non-convex objectives, which is the typical case for DL models. Empirical evaluation across diverse Deep Learning tasks characterized by different types of data heterogeneity demonstrates that
ALSO outperforms both traditional DL approaches and existing DRO methods.

# Using ALSO in practice

To use ALSO outside of this repository, you only need the following:

- [`also.py`](./also.py): the minimal single-file implementation.
- [`example.ipynb`](./example.ipynb): the end-to-end example of training model with ALSO.
- [The section about hyperparameters](#hyperparameters).


---

Table of contents
- [Using ALSO in practice](#using-also-in-practice)
- [Hyperparameters](#hyperparameters)
- [Experiments from the paper](#experiments)
- [How to cite](#how-to-cite)


---

## Hyperparameters

This section covers default hyperparameters and hyperparameter tuning.

**Default hyperparameters**

> [!NOTE]
> This section only provides general advice on default hyperparameter.
> Our experiments indicate ALSO to be robust to hyperparameter tuning in terms of gains over baselines. However, additional tuning can lead to significant performance boost
> However, we strongly recommend to use `pi_reg` and `pi_init` computed as softmax of static weights (i.e., the inverse of ratio of samples of the given class)

| Hyperparameter | $\texttt{ALSO}$       |
| :------------- | :-------------------- |
| mode           | optimistic            |
| alpha          | 1.0                   |
| loss_scale     | n_groups / batch_size |
| lr             | 0.001                 |
| weight_decay   | 0.001                 |
| pi_lr          | 0.001                 |
| pi_decay       | 0.01                  |
| pi_reg         | Uniform               |
| pi_init        | Uniform               |

**Hyperparameter tuning**

During our experiments, we found a **reasonable** search space for hyperparameters. However, one can feel free to change it to make it more suitable for the concrete task.

| Hyperparameter | Search space for $\texttt{ALSO}$ |
| :------------- | :------------------------------- |
| mode           | optimistic                       |
| alpha          | 1.0                              |
| loss_scale     | n_groups / batch_size            |
| lr             | LogUniform[0.00003, 0.001]       |
| weight_decay   | LogUniform[0.0001, 0.1]          |
| pi_lr          | LogUniform[0.00001, 0.001]       |
| pi_decay       | LogUniform[0.001, 1]             |
| pi_reg         | Uniform/Static Weights           |
| pi_init        | Uniform/Static Weights           |

In our experiments we used 100 iterations of the [TPE sampler from Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html). Thus, we suggest to use this amount of iterations. However, in compute limited setup, we suppose that 40-50 iterations should result in a somewhat reasonable.

Additionally, we do not recommend to include `pi_reg` and `pi_init` into the search space, we suggest to simply use Static Weights when reasonable and Uniform elsewise (at least, use the same value for both parameters). However, if one wants to tune this parameters, we recommend to run tuning for each variant potentially with less trials.

# Experiments from the paper

We publish experiments from the [paper](https://arxiv.org/pdf/2508.16734) in the [corresponding folder](./paper).

### TODO
- [ ] Distributed Training Experiments
- [ ] Robust Training to Adversarial Attacks Experiments
- [ ] Ablation Studies Code

# How to cite

```
@article{feoktistov2025aligning,
  title={Aligning distributionally robust optimization with practical deep learning needs},
  author={Feoktistov, Dmitrii and Ignashin, Igor and Veprikov, Andrey and Borovko, Nikita and Bogdanov, Alexander and Chezhegov, Savelii and Beznosikov, Aleksandr},
  journal={arXiv preprint arXiv:2508.16734},
  year={2025}
}
```
