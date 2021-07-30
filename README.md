## Introduction

TBD

## Bare-bones Installation Steps

These steps are the necessary and sufficient steps to download all the dependencies, and load a trained model in the Jupyter notebook.
It does not include steps to generate a synthetic dataset, run in-simulation prediction, or train a model.

Most of the instructions will be done in your console/shell/terminal from your home directory `/home/your-username`, or the repository directory `/home/your-username/carla-collect` denoted as `carla-collect/` for short.

In these instructions, I may tell you to 'do' something, like do `git submodule update`. Do basically means execute a command in the console/shell/terminal.

1) Download the repository from Github:

1.1) `git clone` the repository `carla-collect` to your home directory, or any directory where you keep Github code.

1.2) Got to the directory `carla-collect/` of the repostory, do `git submodule update --init --recursive` to download the Git submodules `carla-collect/Trajectron-plus-plus` and `carla-collect/python-utility`.

2) Install the Python environment and Python dependencies.

2.1) Either use [Anaconda](https://www.anaconda.com/products/individual), or [Python venv](https://docs.python.org/3.6/library/venv.html) to install a virtual environment for Python. You can find a guide to using Anaconda [here](https://docs.anaconda.com/anaconda/user-guide/getting-started/).

If you choose to install Anaconda, then after do:

```
conda create -n trajectron python=3.6
conda activate trajectron
```

2.2) Install the dependencies for `carla-collect/Trajectron-plus-plus` submodule by going to the directory `carla-collect/` and doing:

```
pip install -r Trajectron-plus-plus/requirements.txt
```

2.3) Install the dependencies for `carla-collect` by going to the directory `carla-collect/` and doing:

```
pip install -r requirements.txt
```

3) Install [CARLA Simulator](http://carla.org/). I personally prefer using the **B. Package installation** method from this [guide](https://carla.readthedocs.io/en/0.9.11/start_quickstart/#installation-summary) to install CARLA binaries to my local directory.

3) Edit the `carla-collect/env.sh` as needed. You'll specifically want to "Set CARLA Simulatory directory manually" by setting `CARLA_DIR` and "Enable the Python environment".

Anytime you need to run something in `carla-collect`, please do `source carla-collect/env.sh`. This way your Python language interpreter will "know" the locations of some of the code you need to run for programs.

4) Create a directory `carla-collect/carla_v3-1-dataset`. Download the `.pkl` files for the preprocessed CARLA Synthesized dataset from the Cloud to there.
Cloud Link: <https://drive.google.com/drive/folders/1p8tpx6WEAlTM-yEGoK3fqsCzOasXafHN?usp=sharing>

5) Create a directory `carla-collect/Trajectron-plus-plus/experiments/processed`. Download the `.pkl` files for the preprocessed NuScenes dataset from the Cloud to there.
Cloud Link:

6) Create a directory `carla-collect/Trajectron-plus-plus/experiments/nuScenes/models`. Download the `models_*` directories from the Cloud to there.
Cloud Link: <https://drive.google.com/drive/folders/1EvJXD42KHwXn1lvktILKvv3W4WM4Pxd_?usp=sharing>

7) You're done! Now you can run the Jupyter notebooks `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` by going to `carla-collect/Trajectron-plus-plus/experiments/nuScenes` and doing `source notebook.sh` to start the Jupyter server.

The path `carla-collect/Trajectron-plus-plus/experiments/nuScenes` has the directory `nuScenes`, but it contains models and notebook results for both NuScenes and CARLA experiments.

## Jupyter Notebooks

At this point, I should also discuss what the notebooks in `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` are all about, how to run each notebook individually. There's a lot of stuff here I have to cover so this section is TBD. Be aware that you will encounter bugs in the notebooks as I have not completely cleaned them yet.

You may have to modify code here and there to get some of them to work. The notebooks named `inspect-*.ipynb` and `obtain-*.ipynb` are the ones you most likely care about. The `test-*.pynb` notebooks are what I use for prototyping.

## Synthesizing CARLA Dataset

TBD

## Training/Evaluating Trajectron++ Model

TBD

## Training/Evaluating ESP Model

Very TBD

## In-Simulation Prediction

TBD
