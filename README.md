## Introduction

TBD

## Technical aspects of versioning

Techical goals:

- Ideally, both `carla-collect` and Trajectron++ should be upgraded to Python 3.8 and Torch 1.11 and the latest libraries.
- Datasets for Trajectron++ should not be persisted Python objects (e.g. use `dill`). Instead use HDF5 or similar storage.
- Ideally, use CARLA 0.9.13, but it requies 8 GB GPU memory. CARLA 0.9.11 works as an alternative, but it's not possible to fix autopilot car routes using this version.

Observations:

- CARLA 0.9.11 provides a client egg library for Python 3.7, but it seems to work with Python 3.8.
- CARLA 0.9.13 provides a client wheel installable from PyPI for Python 3.6-8.
- `scipy==1.3.1` : requires blas unlike later versions, according to [source](https://github.com/scipy/scipy/issues/9005) you need to do `sudo apt-get install gfortran libopenblas-dev liblapack-dev` first.
- Trajectron++ is currently trained using Python 3.6 Torch 1.4, but it works with Python 3.8. Model seems trainable with Torch 1.7.1 and trained models seem runnable using Torch 1.11.
- Pickled objects using `dill` seem to be unpicklable in environments with a different version of `numpy`, etc and `dill`.
- Installed PyTorch binaries should come with CUDA, cuDNN.
- NVIDIA GeFroce RTX 3050 seems to cause runtime error when using CUDA 10.1 e.g. `torch==1.4.0`, `torch` with CUDA 11 binaries seems to work e.g. Torch 1.7.1 and 1.11.
- ComputeCanada offer Python wheels for most packages but not all packages. Install higher version provided wheels if available, and download packages from the internet for non-offered packages.

## Installation

These steps are the necessary and sufficient steps to download all the dependencies, and load a trained model in the Jupyter notebook.
It does not include steps to generate a synthetic dataset, run in-simulation prediction, or train a model.

Most of the instructions will be done in your console/shell/terminal from your home directory `/home/your-username`,
or the repository directory `/home/your-username/carla-collect` denoted as `carla-collect/` for short.

In these instructions, I may tell you to 'do' something, like do `git submodule update`. Do basically means execute a command in the console/shell/terminal.

**Step 1**. Install system dependencies.

```
# dependencies for CARLA
sudo apt install libomp5
# dependencies for building Python libraries when calling pip install
sudo apt install build-essential gfortran libopenblas-dev liblapack-dev
```

**Step 2**.
Download the repository from Github:

**Step 2.1**.
`git clone` the repository `carla-collect` to your home directory,
or any directory where you keep Github code.

**Step 2.2**.
Got to the directory `carla-collect/` of the repostory,
do `git submodule update --init --recursive` to download the Git submodules `carla-collect/Trajectron-plus-plus` and `carla-collect/python-utility`.

**Step 3**.
Install the Python environment and Python dependencies.

**Step 3.1**.
Use [Python venv](https://docs.python.org/3.6/library/venv.html) to install a virtual environment for Python 3.7
(Python 3.8 causes incompatibility with CPLEX).
The Ubuntu distribution may not provide this version by default. If so, please install the deadsnakes PPA:

```
# setup python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt install python3.7 python3.7-dev python3.7-venv
python3.7 -m venv 
```

Afterwards, go to the directory `carla-collect/`, activate the virtual environment, and update this environment by doing

```
python3 -m venv py37trajectron
source py37trajectron/bin/activate
pip install --upgrade pip
pip install wheel
```

**Step 3.2**.
Install the dependencies for `carla-collect/Trajectron-plus-plus` submodule by going to the directory `carla-collect/` and doing:

```
pip install -r Trajectron-plus-plus/requirements.txt
```

**Step 3.3**.
Install PyTorch version 1.7.1 (see [previous PyTorch versions](https://pytorch.org/get-started/previous-versions/)).
The binaries should depend on your GPU card's CUDA compatibility, but most modern GPUs doing the below is sufficient:

```
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.4**.
Install the dependencies for `carla-collect` by going to the directory `carla-collect/` and doing:

```
pip install -r requirements.txt
```

**Step 4**.
Install [CARLA Simulator](http://carla.org/).
I used the **B. Package installation** method from this [guide](https://carla.readthedocs.io/en/0.9.11/start_quickstart/#installation-summary) to download precompiled CARLA binaries to my local directory.
`carla-collect` is set up to use CARLA versions 0.9.11 or 0.9.13.
Ideally, you want to run CARLA 0.9.13 since it has the necessary Python API to run some experiments, but it takes more GPU memory to run.
I installed CARLA in the directory `/home/your-username/src`. If you install it somewhere else please modify the following instructions accordingly.

**Step 4.1**.
Install the Python client for CARLA. 

- If you installed CARLA version 0.9.13, then `pip install carla==0.9.13`.
- If you installed CARLA version 0.9.11, simply add
`$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg` to your `PYTHONPATH`.
I've provided a way to do this automatically using the script `py37.env.sh` described later.

**Step 5**.
Install IBM ILOG CPLEX Optimization Studio V12.10.0.
It should be installed to `/opt/ibm/ILOG/CPLEX_Studio1210`.
The script `py37.env.sh` should automatically export the installation path so it can be discoverable by the `docplex` library used in `carla-collect`.

**Step 6**.
Configure the startup script `carla-collect/py37.env.sh`.

- If you installed CARLA version 0.9.13, uncomment `export CARLANAME=carla-0.9.13`.
- If you installed CARLA version 0.9.11, uncomment `export CARLANAME=carla-0.9.11`.

**Step 6.1**.
Double check that the path to CPLEX and CARLA is set correctly (i.e. `CPLEX_STUDIO_DIR1210` and `CARLA_DIR`).

**Step 6.2**.
Anytime you need to run something in `carla-collect`, please do `source carla-collect/py37.env.sh`.
This way your Python language interpreter will "know" the locations of some of the code you need to run for programs.

**Step 7**.
Download the dataset and models.

**Step 7.1**.
Download the `.pkl` files for the preprocessed CARLA v3-1-2 dataset from Google Drive into the directory
`carla-collect/carla_dataset_v3-1-2`.
This dataset is synthesized from CARLA 0.9.11 using `synthesize.py`.

Cloud Link: <https://drive.google.com/drive/folders/1DJZUP-V6puxPHxVU9suT0svTpnX0Qt4v?usp=sharing>

**Step 7.2**.
Download the `.pkl` files for the preprocessed CARLA v4-2-1 dataset from Google Drive into the directory
`carla-collect/carla_dataset_v3-1-2`.
This dataset is synthesized from CARLA 0.9.13 using `carla-collect/synthesize.py`.
The vehicles in the dataset behave quite differently because the vehicles were overhauled.

Cloud Link: <https://drive.google.com/drive/folders/1PR587zadQ_C3jHT3o_56XGBIlgXVCZbR?usp=sharing>

**Step 7.3**.
Download the `.pkl` files for the preprocessed nuScenes dataset from Google Drive into the directory
`carla-collect/Trajectron-plus-plus/experiments/processed`.
This dataset is synthesized from the nuScenes dataset `carla-collect/Trajectron-plus-plus/python process_data.py`.

Cloud Link: <https://drive.google.com/drive/folders/1mKp1MLgGMdNaOXLOoD81jgunbbqdDwlU?usp=sharing>

**Step 7.4**.
Download the models from Google Drive

- `models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8`
trained on the CARLA v3-1-2 dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210816`
- `models_27_Mar_2022_04_52_55_carla_v4-2-1_base_distmapV4_modfm_K15_ph8`
trained on the CARLA v4-2-1 dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20220326`
- `models_19_Mar_2021_22_14_19_int_ee_me_ph8`
trained on the nuScenes based dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210621`

Cloud Link: <https://drive.google.com/drive/folders/1EvJXD42KHwXn1lvktILKvv3W4WM4Pxd_?usp=sharing>

**Step 8**. You're done! to un the Jupyter notebooks `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` by going to `carla-collect/Trajectron-plus-plus/experiments/nuScenes` and doing `source notebook.sh` to start the Jupyter server.

The path `carla-collect/Trajectron-plus-plus/experiments/nuScenes` has the directory `nuScenes`, but it contains models and notebook results for both NuScenes and CARLA experiments.

## Jupyter Notebooks

At this point, I should also discuss what the notebooks in `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` are all about, how to run each notebook individually. There's a lot of stuff here I have to cover so this section is TBD. Be aware that you will encounter bugs in the notebooks as I have not completely cleaned them yet.

You may have to modify code here and there to get some of them to work. The notebooks named `inspect-*.ipynb` and `obtain-*.ipynb` are the ones you most likely care about. The `test-*.pynb` notebooks are what I use for prototyping.

## Running CARLA and Synthesizing CARLA Dataset

Collect data via

```
source collect_Mix.sh 2>&1 | tee out/log.txt
```

TBD

## Training/Evaluating Trajectron++ Model

TBD

## Training/Evaluating ESP Model

Very TBD

## In-Simulation Prediction

TBD

## Work In Progress
