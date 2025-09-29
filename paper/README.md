# Experiments from "Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"

This folder contains the code for experiments reproduction from the paper ["Aligning Distributionally Robust Optimization with Practical Deep Learning Needs"](https://arxiv.org/abs/2508.16734). Each experimental subsection is placed in its own folder:

- `Learning From Unbalanced Data`: Section 6.1
- `Tabular DL`: Section 6.2
- `Split Data`: Section 6.5

To reproduce the results, you should setup the environment and then visit the folder corresponding to the experiment you are interested in.

## Set up the environment

Install any of the following tools, and follow the remaining instructions.

| Tool                                                                                           | Supported OS | Supported devices |
| :--------------------------------------------------------------------------------------------- | :----------- | :---------------- |
| [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) | Linux        | GPU               |
| [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)           | Linux        | GPU               |
| [Conda](https://conda-forge.org/download/)                                                     | Linux        | GPU               |


With Micromamba:

```
micromamba create -f environment.yaml
micromamba activate also
```

With Mamba:

```
mamba create -f environment.yaml
mamba activate also
```

With Conda:

```
conda create -f environment.yaml -n tabm
conda activate also
```
