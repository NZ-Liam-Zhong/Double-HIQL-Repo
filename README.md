# We use the same environment as OGBench.


# Overview

OGBench is a benchmark designed to facilitate algorithms research in offline goal-conditioned reinforcement learning (RL),
offline unsupervised RL, and offline RL.
See the [project page](https://seohong.me/projects/ogbench/) for videos and more details about the environments, tasks, and datasets.

### Features

- **8 types** of realistic and diverse environments ([videos](https://seohong.me/projects/ogbench/)):
  - **Locomotion**: PointMaze, AntMaze, HumanoidMaze, and AntSoccer.
  - **Manipulation**: Cube, Scene, and Puzzle.
  - **Drawing**: Powderworld.
- **85 datasets** covering various challenges in offline goal-conditioned RL.
- **410 tasks** for standard (i.e., non-goal-conditioned) offline RL.
- Support for both **pixel-based** and **state-based** observations.
- **Clean, well-tuned reference implementations** of 6 offline goal-conditioned RL algorithms
(GCBC, GCIVL, GCIQL, QRL, CRL, and HIQL) based on JAX.
- **Fully reproducible** scripts for [the entire benchmark table](impls/hyperparameters.sh)
and [datasets](data_gen_scripts/commands.sh).
- `pip`-installable, easy-to-use APIs based on Gymnasium.
- No major dependencies other than MuJoCo.



# Quick Start

### Installation （这个章节安装环境即可完成）

OGBench can be easily installed via PyPI:

```shell
pip install ogbench
```

It requires Python 3.8+ and has only three dependencies: `mujoco >= 3.1.6`, `dm_control >= 1.0.20`,
and `gymnasium`.

To use OGBench for **offline goal-conditioned RL**,
go to [this section](#usage-for-offline-goal-conditioned-rl).
To use OGBench for **standard (non-goal-conditioned) offline RL**,
go to [this section](#usage-for-standard-non-goal-conditioned-offline-rl).

Our reference implementations require Python 3.9+ and additional dependencies, including `jax >= 0.4.26`.
To install these dependencies, run:

```shell
cd impls
pip install -r requirements.txt
```

By default, it uses the PyPI version of OGBench.
If you want to use a local version of OGBench (e.g., for training methods on modified environments),
run instead `pip install -e ".[train]"` in the root directory.


# Reproducing Datasets（这部分是生成数据集的代码）

We provide the full scripts and exact command-line flags used to produce all the datasets in OGBench.
The scripts are provided in the `data_gen_scripts` directory.

### Installation

Data-generation scripts for locomotion environments require Python 3.9+ and additional dependencies,
including `jax >= 0.4.26`, to train and load expert agents.
For manipulation and drawing environments, no additional dependencies are required.
To install the necessary dependencies for locomotion environments, run the following command in the root directory:
```shell
pip install -e ".[train]"
```

This installs the same dependencies as the reference implementations, but in the editable mode (`-e`).

### Reproducing datasets

To reproduce datasets, you can run the scripts in the `data_gen_scripts` directory.
For locomotion environments, you need to first download the expert policies.
We provide the exact command-line flags used to produce the datasets in [commands.sh](data_gen_scripts/commands.sh).
Here is an example of how to reproduce a dataset for the `antmaze-large-navigate-v0` task:

```shell
cd data_gen_scripts
# Download the expert policies for locomotion environments (not required for other environments).
wget https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz
tar xf experts.tar.gz && rm experts.tar.gz
# Create a directory to save datasets.
mkdir -p data
# Add the `impls` directory to PYTHONPATH.
# Alternatively, you can move the contents of `data_gen_scripts` to `impls` instead of setting PYTHONPATH.
export PYTHONPATH="../impls:${PYTHONPATH}"  
# Generate a dataset for `antmaze-large-navigate-v0`.
python generate_locomaze.py --env_name=antmaze-large-v0 --save_path=data/antmaze-large-navigate-v0.npz
```

