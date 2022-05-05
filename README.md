## Introduction

Work to extend the [CDC paper](https://ieeexplore.ieee.org/document/9455144) and the deep learning model Trajectron++, to motion planning in CARLA Simulator.

## Installation

These steps are the necessary and sufficient steps to download all the dependencies, and load a trained model in the Jupyter notebook.
It does not include steps to generate a synthetic dataset, run in-simulation prediction, or train a model.

Most of the instructions will be done in your console/shell/terminal from your home directory `/home/your-username`,
or the repository directory `/home/your-username/carla-collect` denoted as `carla-collect/` for short.

In these instructions, I may tell you to 'do' something, like do `git submodule update`. Do basically means execute a command in the terminal (aka. console, shell).

**Step 1**.
Install system dependencies.

```
sudo apt update
sudo apt upgrade
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
I strongly suggest NOT using Anaconda/Miniconda as it does not work well with HPCs i.e. Compute Canada,
and adds extra complexity to the project over the long term.

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
source py37trajectron/bin/activate
pip install -r Trajectron-plus-plus/requirements.txt
```

**Step 3.3**.
Install PyTorch version 1.7.1 (see [previous PyTorch versions](https://pytorch.org/get-started/previous-versions/)).
The binaries should depend on your GPU card's CUDA compatibility, but most modern GPUs doing the below is sufficient:

```
source py37trajectron/bin/activate
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.4**.
Install the dependencies for `carla-collect` by going to the directory `carla-collect/` and doing:

```
source py37trajectron/bin/activate
pip install -r requirements.txt
```

**Step 4**.
Install [CARLA Simulator](http://carla.org/).
I used the **B. Package installation** method from this [guide](https://carla.readthedocs.io/en/0.9.11/start_quickstart/#installation-summary) to download precompiled CARLA binaries to my local directory.
`carla-collect` is set up to use CARLA versions 0.9.11 or 0.9.13.
Ideally, you want to run CARLA 0.9.13 since it has the necessary Python API to run some experiments, but it takes more GPU memory to run.
I installed CARLA 0.9.11 in directory `/home/your-username/src/carla-0.9.11` and 0.9.13 in `/home/your-username/src/carla-0.9.13`.
If you install it somewhere else please modify the following instructions accordingly. To run CARLA simply do something like

```
cd /home/your-username/src/carla-0.9.11
./CarlaUE4.sh
```

**Step 4.1**.
Install the Python client for CARLA. 

- If you installed CARLA version 0.9.13, then `pip install carla==0.9.13`.
- If you installed CARLA version 0.9.11, simply add
`$CARLA_DIR/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg` to your `PYTHONPATH`.
I've provided a way to do this automatically using the script `py37.env.sh` described later.

**Step 5 (optional)**.
Install IBM ILOG CPLEX Optimization Studio V12.10.0.
Skip this step if you don't want to run [in-simulation experiments](#in-simulation-experiments).
It should be installed to `/opt/ibm/ILOG/CPLEX_Studio1210`.
The script `py37.env.sh` should automatically export the installation path so it can be discoverable by the `docplex` library used in `carla-collect`.

**Step 6**.
Edit the startup BASH script `carla-collect/py37.env.sh`.

- If you installed CARLA version 0.9.13, uncomment `export CARLANAME=carla-0.9.13`.
- If you installed CARLA version 0.9.11, uncomment `export CARLANAME=carla-0.9.11`.

**Step 6.1**.
Double check that the path to CPLEX and CARLA is set correctly (i.e. `CPLEX_STUDIO_DIR1210` and `CARLA_DIR`).

**Step 6.2**.
Anytime you need to run something in `carla-collect`, please run
the BASH script in your current terminal environment by doing
`source carla-collect/py37.env.sh`.
This way your Python language interpreter will automatically "know"
the locations of some of the codebase you need to run for programs
without you having to manually set up your terminal environment.

**Step 7 (optional)**.
Download the datasets.
Skip this step if you only want to run [in-simulation experiments](#in-simulation-experiments) or synthesize your own dataset.
It's necessary if you want to run the Jupyter notebooks in `carla-collect/Trajectron-plus-plus`.

**Step 7.1**.
Download the `.pkl` files for the preprocessed CARLA v3-1-2 dataset from Google Drive into the directory
`carla-collect/carla_dataset_v3-1-2`.
This dataset is synthesized from CARLA 0.9.11 using `synthesize.py`.

Cloud Link: <https://drive.google.com/drive/folders/1DJZUP-V6puxPHxVU9suT0svTpnX0Qt4v?usp=sharing>

**Step 7.2**.
Download the `.pkl` files for the preprocessed CARLA v4-2-1 dataset from Google Drive into the directory
`carla-collect/carla_dataset_v4-2-1`.
This dataset is synthesized from CARLA 0.9.13 using `carla-collect/synthesize.py`.
The vehicles in the dataset behave quite differently because the vehicles were overhauled.

Cloud Link: <https://drive.google.com/drive/folders/1PR587zadQ_C3jHT3o_56XGBIlgXVCZbR?usp=sharing>

Note, you need Python 3.8 if you want to load this dataset to do model training, etc.
Redo step 3 with Python 3.8 and create a virtual environment py38trajectron with the same packages installed.

**Step 7.3**.
Download the `.pkl` files for the preprocessed nuScenes dataset from Google Drive into the directory
`carla-collect/Trajectron-plus-plus/experiments/processed`.
This dataset is synthesized from the nuScenes dataset `carla-collect/Trajectron-plus-plus/python process_data.py`.

Cloud Link: <https://drive.google.com/drive/folders/1mKp1MLgGMdNaOXLOoD81jgunbbqdDwlU?usp=sharing>

**Step 8**.
Download the models from Google Drive

- `models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8`
trained on the CARLA v3-1-2 dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210816`
- `models_27_Mar_2022_04_52_55_carla_v4-2-1_base_distmapV4_modfm_K15_ph8`
trained on the CARLA v4-2-1 dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20220326`
Even though this model was trained using a Python 3.8 dataset, it is still compatible with 3.7. 
- `models_19_Mar_2021_22_14_19_int_ee_me_ph8`
trained on the nuScenes based dataset into the directory
`carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210621`

Cloud Link: <https://drive.google.com/drive/folders/1EvJXD42KHwXn1lvktILKvv3W4WM4Pxd_?usp=sharing>

**Step 9**.
You're done!

## Documentation

Development, codebase, and advanced documentation is found by reading `carla-collect/build/html/index.html`.
To rebuild this documentation from source `carla-collect/source`, do `source py37.env.sh; make html`.

## In-Simulation Experiments

You need to start CARLA to run these, so do that first.
In `carla-collect`, all tests are implemented using the testing library [pytest](`https://docs.pytest.org/`).
Please list all tests/experiments by doing:

```
source py37.env.sh
pytest tests/Hz20 --collect-only
```

Here are the most relevant tests:

```
pytest tests/Hz20/test_planner.py::test_Town03_scenario[v8-scene4_ov2_gap55-ph8_step1_ncoin1_r_np5000]
```

Runs a scenario with a T-intersection and 2 OVs. The EV attempts to make a left turn while avoiding incoming OVs. 
This tests saves plots to `carla-collect/out` for predictions at each MPC step,
plots for predicted/actual trajectory at each MPC step. 

```
pytest tests/Hz20/test_montecarlo.py::test_Town03_scenario[v8-scene4_ov2_gap55-ph8_step1_ncoin1_r_np5000]
```

Runs 100 trials on the same scenario as above,
printing metrics including fraction of success, raction of infeasibility,
and mean number of steps in simulation to the terminal.

```
pytest tests/Hz20/test_clusters.py::test_Town03_scenario[scene3_veh4-ph6_np5000]
```

Runs autopilot OVs without the EV.
For each OV at each 0.5s interval (MPC steptime) we plot convex hulls containing predictions to `carla-collect/out`.
Each plot renders a convex hull for each predicted step in the future from the initial time the prediction is made.
Similarly,

```
pytest tests/Hz20/test_montecarlo.py::test_Town03_scenario[LABEL]
pytest tests/Hz20/test_planner.py::test_Town03_scenario[LABEL]
```

These tests are the same as the preceding ones where LABEL = `MIDLEVEL-SCENARIO_PARAMS-CTRL_PARAMS` and

- MIDLEVEL is one of
  + `v8` Planner solving original chance constrained problem in MPC.
    Optimizes a single trajectory avoiding collision at each step.
  + `v9` Planner solving contingency planning chance constrainted problem in MPC.
    Optimizes multiple trajectories where at least one avoids collision at each step.
- SCENARIO_PARAMS is one of
  + `scene3_ov4_gap60` big 5-intersection with 4 OVs. There is a 60m gap between the 2 OVs in the front and 2 OVs in the back.
  + `scene4_ov1_accel` small T-intersection with 1 OV. OV is positioned so EV accelerates to overcome OV.
  + `scene4_ov1_brake` small T-intersection with 1 OV. OV is positioned so EV brakes.
  + `scene4_ov2_gap55` small T-intersection with 2 OVs and 55m gap between them.
- CTRL_PARAMS is one of
  + `ph8_step1_ncoin1_r_np5000` Planning horizon 8; call MPC controller at each MPC step; coinciding horizon 1; randomized contingency planning if applicable; number of predictions 5000.
  + `ph6_step1_ncoin1_r_np5000` Same as above, but with planning horizon 6.

## Synthesizing CARLA Dataset

Collect data to train Trajectron++ using `carla_collect/synthesize.py`.
Data consists of multiple scenes. Each scene contains and overhead map and vehicles nodes.

**Step 1**.
Start CARLA. While CARLA is running, run the
BASH scripts provided for convenience.

```
source py37.env.sh
# to generate Trajectron++ data from maps Town03, Town04, ..., Town10HD
source collect_Mix.sh 2>&1 | tee out/log.txt
# to generate Trajectron++ data from map in Town03 only
source collect_Town03.sh 2>&1 | tee out/log.txt
```

Using pipe `|` and `tee` ensures the output of the program shows up in the terminal for debugging, and is logged to a file.

**Step 2**.
Start CARLA. While CARLA is running, do `python cache_carla_maps.py` to save extract specific map data from CARLA to disk to be used later.

**Step 3**.
Create (train, validation, test) sets from the data.
`carla-collect/split_dataset.py` can create cross-validation splits be first creating `--n-groups` groups,
selects a test set from one group, selects a validation set from another group, and selects a training set from the rest of the groups.

```
# get usage
python split_dataset.py -h
# save one split with 500 training, 60 validation, and 60 test samples.
python split_dataset.py --data-files $(ls out/*.pkl) --n-groups 6 --n-splits 1 --n-train 500 --n-val 60 --n-test 60 --label YOUR_LABEL_HERE
# use weighted sampling
python split_dataset.py --data-files $(ls out/*.pkl) --n-groups 6 --n-splits 1 --n-train 500 --n-val 60 --n-test 60 --weighted-resampling --label YOUR_LABEL_HERE
```

Additional options:

- `--weighted-resampling` use weighted sampling when sampling a set from groups instead of uniform sampling. 
  Scenes with more nodes turning or crossing intersections are weighted higher. 

**Step 4**.
Trajectron++ preprocessing code frequency weights the vehicle nodes in training data so that nodes of turning vehicles are sampled more often
(i.e. sampling at each stochastic gradient descent step in training loop).
I found that updating the node weights improved model performance
(e.g. you probably care more about vehicle behavior at intersections).
You may want to update the vehicle node weights using `carla-collect/modify_fm.py`.
First get breakdown of the node counts by doing

```
python modify_fm.py --data-path out/YOUR_LABEL_HERE_split1_train.pkl --dry-run
```

The terminal will display a breakdown with the classes:

- `complete_intersection`: vehicle enters and leaves the intersection.
- `significant_at_intersection`: vehicle is near an intersection and is not idle.
- `stopped_at_intersection`: vehicle is idle near an intersection.
- `other_at_intersection`: vehicle is doing something else at an intersection.
- `turn_at_other`: vehicle is turning. Not at an intersection.
- `significant_at_other`: vehicle is not turning or idle. Not at an intersection.
- `stopped_at_other`: vehicle is idle. Not at an intersection.
- `other_at_other`: vehicle is doing something else. Not at an intersection.

Modify `carla-collect/modifier.json` with the frequencies you want for the nodes in the respective classes (2:1 means twice as freqently sampled), and update the training set.

```
python modify_fm.py --data-path out/YOUR_LABEL_HERE_split1_train.pkl --modifier modifier.json -v
```

This creates a new training set `out/YOUR_LABEL_HERE_split1_train_modfm.pkl` with updated weights.

## Training/Evaluating Trajectron++ Model

The training code for Trajectron++ can be found in the directory
`carla-collect/Trajectron-plus-plus/trajectron`.
If you completed **Step 7** of installation then there are 3 training BASH scripts you can use to train the model:

- `carla_v3-1-2.train_base_distmapV4.modfm.K15.ph8.sh` train model on dataset synthesized by CARLA 0.9.11.
- `carla_v4-2-1.train_base_distmapV4.modfm.K15.ph8.sh` train model on dataset synthesized by CARLA 0.9.13.
- `nuScenes.train_dynmap.ph8.sh` train model on nuScenes preprocessed dataset.

Simply call them like so `source SCRIPT.sh`.
Feel free to modify these scripts to train a model on your own dataset.
See the README in `carla-collect/Trajectron-plus-plus` for more about the model.

## Jupyter Notebooks

Run the Jupyter notebooks `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` by going to `carla-collect/Trajectron-plus-plus/experiments/nuScenes` and doing `jupyter notebook` to start the Jupyter server.

The path `carla-collect/Trajectron-plus-plus/experiments/nuScenes` has the directory `nuScenes`, but it contains models and notebook results for both NuScenes and CARLA experiments.

<!-- At this point, I should also discuss what the notebooks in `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` are all about, how to run each notebook individually. There's a lot of stuff here I have to cover so this section is TBD. Be aware that you will encounter bugs in the notebooks as I have not completely cleaned them yet. -->

You may have to modify code here and there to get some of them to work. The notebooks named `inspect-*.ipynb` and `obtain-*.ipynb` are the ones you most likely care about. The `test-*.pynb` notebooks are what I use for prototyping.

<!-- Start CARLA. While CARLA is running, do `python cache_carla_maps.py` to save extract specific map data from CARLA to disk to be used later. -->

