## Introduction

TBD

## Bare-bones Installation Steps

These steps are the necessary and sufficient steps to download all the dependencies, and load a trained model in the Jupyter notebook.
It does not include steps to generate a synthetic dataset, run in-simulation prediction, or train a model.

Most of the instructions will be done in your console/shell/terminal from your home directory `/home/your-username`, or the repository directory `/home/your-username/carla-collect` denoted as `carla-collect/` for short.

In these instructions, I may tell you to 'do' something, like do `git submodule update`. Do basically means execute a command in the console/shell/terminal.

**Step 1**. Download the repository from Github:

**Step 1.1**. `git clone` the repository `carla-collect` to your home directory, or any directory where you keep Github code.

**Step 1.2**. Got to the directory `carla-collect/` of the repostory, do `git submodule update --init --recursive` to download the Git submodules `carla-collect/Trajectron-plus-plus` and `carla-collect/python-utility`.

**Step 2**. Install the Python environment and Python dependencies.

**Step 2.1**. Either use [Anaconda](https://www.anaconda.com/products/individual), or [Python venv](https://docs.python.org/3.6/library/venv.html) to install a virtual environment for Python. You can find a guide to using Anaconda [here](https://docs.anaconda.com/anaconda/user-guide/getting-started/).

If you choose to install Anaconda, then after do:

```
conda create -n trajectron python=3.6
conda activate trajectron
```

**Step 2.2**. Install the dependencies for `carla-collect/Trajectron-plus-plus` submodule by going to the directory `carla-collect/` and doing:

```
pip install -r Trajectron-plus-plus/requirements.txt
```

**Step 2.3**. Install the dependencies for `carla-collect` by going to the directory `carla-collect/` and doing:

```
pip install -r requirements.txt
```

**Step 3**. Install [CARLA Simulator](http://carla.org/). I personally prefer using the **B. Package installation** method from this [guide](https://carla.readthedocs.io/en/0.9.11/start_quickstart/#installation-summary) to install CARLA binaries to my local directory.

**Step 4**. Edit the `carla-collect/env.sh` as needed. You'll specifically want to "Set CARLA Simulatory directory manually" by setting `CARLA_DIR` and "Enable the Python environment".

Anytime you need to run something in `carla-collect`, please do `source carla-collect/env.sh`. This way your Python language interpreter will "know" the locations of some of the code you need to run for programs.

**Step 5**. Create a directory `carla-collect/carla_v3-1-dataset`. Download the `.pkl` files for the preprocessed CARLA Synthesized dataset from the Cloud to there.  
Cloud Link: <https://drive.google.com/drive/folders/1p8tpx6WEAlTM-yEGoK3fqsCzOasXafHN?usp=sharing>

**Step 6**. Create a directory `carla-collect/Trajectron-plus-plus/experiments/processed`. Download the `.pkl` files for the preprocessed NuScenes dataset from the Cloud to there.  
Cloud Link: <https://drive.google.com/drive/folders/1mKp1MLgGMdNaOXLOoD81jgunbbqdDwlU?usp=sharing>

**Step 7**. Create a directory `carla-collect/Trajectron-plus-plus/experiments/nuScenes/models`. Download the `models_*` directories from the Cloud to there.  
Cloud Link: <https://drive.google.com/drive/folders/1EvJXD42KHwXn1lvktILKvv3W4WM4Pxd_?usp=sharing>

**Step 8**. You're done! Now you can run the Jupyter notebooks `carla-collect/Trajectron-plus-plus/experiments/nuScenes/notebooks` by going to `carla-collect/Trajectron-plus-plus/experiments/nuScenes` and doing `source notebook.sh` to start the Jupyter server.

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

### CPLEX Runtime is slow?

```
(trajectron-cplex) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
======================================== test session starts ========================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                    

tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000] 
Loading from /home/fireofearth/code/robotics/carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210816/models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8/model_registrar-20.pt
Loaded!


------------------------------------------- live log call -------------------------------------------
INFO     numexpr.utils:utils.py:145 Note: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO     numexpr.utils:utils.py:157 NumExpr defaulting to 8 threads.
Version identifier: 12.10.0.0 | 2019-11-26 | 843d4de
CPXPARAM_Read_DataCheck                          1
Tried aggregator 1 time.
Reduced MIQP has 176 rows, 80 columns, and 1344 nonzeros.
Reduced MIQP has 64 binaries, 0 generals, 0 SOSs, and 0 indicators.
Reduced MIQP objective Q matrix has 128 nonzeros.
Presolve time = 0.00 sec. (0.38 ticks)
Probing time = 0.00 sec. (0.02 ticks)
Tried aggregator 1 time.
Reduced MIQP has 176 rows, 80 columns, and 1344 nonzeros.
Reduced MIQP has 64 binaries, 0 generals, 0 SOSs, and 0 indicators.
Reduced MIQP objective Q matrix has 128 nonzeros.
Presolve time = 0.00 sec. (0.37 ticks)
Classifier predicts products in MIQP should be linearized.
Probing time = 0.00 sec. (0.02 ticks)
Clique table members: 32.
MIP emphasis: balance optimality and feasibility.
MIP search method: dynamic search.
Parallel mode: deterministic, using up to 12 threads.

Warning:  Barrier optimality criterion affected by large objective shift.
Root relaxation solution time = 0.01 sec. (2.85 ticks)

        Nodes                                         Cuts/
   Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap

      0     0       -0.0000    32                     -0.0000       11         
      0     2       -0.0000    32                     -0.0000       11         
Elapsed time = 0.05 sec. (20.52 ticks, tree = 0.02 MB, solutions = 0)
*   212+   16                         1640.8158        0.0000           100.00%
*   276    11      integral     0     1586.6985     1586.6985     2268    0.00%
*   284     7      integral     0     1586.6985     1586.6985     2780    0.00%
*   289     7      integral     0     1586.6985     1586.6985     2958    0.00%

Root node processing (before b&c):
  Real time             =    0.05 sec. (20.28 ticks)
Parallel b&c, 12 threads:
  Real time             =    0.11 sec. (19.01 ticks)
  Sync time (average)   =    0.10 sec.
  Wait time (average)   =    0.00 sec.
                          ------------
Total (root+branch&cut) =    0.16 sec. (39.29 ticks)
INFO     root:util.py:67 code profile of __compute_prediction_controls
INFO     root:util.py:68          180521 function calls (170267 primitive calls) in 0.900 seconds

   Ordered by: cumulative time
   List reduced from 2654 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.900    0.900 __init__.py:511(__compute_prediction_controls)
        1    0.001    0.001    0.470    0.470 __init__.py:436(do_highlevel_control)
        1    0.000    0.000    0.422    0.422 __init__.py:226(do_prediction)
        1    0.000    0.000    0.370    0.370 scene.py:437(get_scene)
        1    0.001    0.001    0.235    0.235 trajectron_scene.py:454(process_scene)
        1    0.000    0.000    0.217    0.217 model.py:354(__init__)
        1    0.099    0.099    0.213    0.213 trajectron_scene.py:351(__process_carla_scene)
        2    0.000    0.000    0.190    0.095 environment.py:60(__init__)
        1    0.000    0.000    0.167    0.167 model.py:4671(solve)
        1    0.000    0.000    0.165    0.165 model.py:4749(_solve_local)
        1    0.000    0.000    0.161    0.161 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.161    0.161 __init__.py:1282(solve)
        1    0.000    0.000    0.161    0.161 _procedural.py:689(mipopt)
        1    0.000    0.000    0.161    0.161 _pycplex.py:1532(CPXXmipopt)
        1    0.160    0.160    0.161    0.161 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
        1    0.000    0.000    0.146    0.146 model.py:176(_make_environment)
        1    0.000    0.000    0.146    0.146 environment.py:426(get_default_env)
        1    0.000    0.000    0.146    0.146 environment.py:421(make_new_configured_env)
        1    0.000    0.000    0.134    0.134 scene.py:402(__checkpoint)
        1    0.038    0.038    0.111    0.111 scene.py:309(__process_lidar_snapshots)
     42/4    0.000    0.000    0.106    0.026 <frozen importlib._bootstrap_external>:672(exec_module)
     97/5    0.000    0.000    0.104    0.021 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
     45/7    0.000    0.000    0.102    0.015 <frozen importlib._bootstrap>:966(_find_and_load)
     45/7    0.000    0.000    0.102    0.015 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
     43/6    0.000    0.000    0.100    0.017 <frozen importlib._bootstrap>:651(_load_unlocked)
     43/4    0.000    0.000    0.099    0.025 {built-in method builtins.exec}
        2    0.000    0.000    0.099    0.050 environment.py:216(auto_configure)
        2    0.000    0.000    0.099    0.050 environment.py:323(check_cplex)
        3    0.000    0.000    0.099    0.033 environment.py:227(get_cplex_module)
        1    0.000    0.000    0.098    0.098 environment.py:276(load_cplex_from_cos_root)
        1    0.000    0.000    0.098    0.098 environment.py:249(load_cplex)
        1    0.000    0.000    0.093    0.093 __init__.py:33(<module>)
  512/438    0.000    0.000    0.092    0.000 <frozen importlib._bootstrap>:997(_handle_fromlist)
    21/16    0.000    0.000    0.091    0.006 {built-in method builtins.__import__}
        2    0.000    0.000    0.091    0.045 platform.py:834(architecture)
        2    0.000    0.000    0.091    0.045 platform.py:798(_syscmd_file)
       42    0.000    0.000    0.088    0.002 <frozen importlib._bootstrap_external>:743(get_code)
        2    0.000    0.000    0.086    0.043 subprocess.py:608(__init__)
        2    0.000    0.000    0.086    0.043 subprocess.py:1228(_execute_child)
        1    0.000    0.000    0.085    0.085 aborter.py:12(<module>)
        1    0.000    0.000    0.084    0.084 __init__.py:15(<module>)
       30    0.000    0.000    0.080    0.003 <frozen importlib._bootstrap_external>:735(source_to_code)
       30    0.080    0.003    0.080    0.003 {built-in method builtins.compile}
        1    0.000    0.000    0.066    0.066 model.py:342(_new_engine)
        1    0.000    0.000    0.066    0.066 model.py:1180(_make_new_engine_from_agent)
        1    0.000    0.000    0.066    0.066 engine_factory.py:90(new_engine)
        1    0.000    0.000    0.066    0.066 cplex_engine.py:463(__init__)
        1    0.000    0.000    0.066    0.066 cplex_adapter.py:49(__init__)
        1    0.000    0.000    0.052    0.052 prediction.py:19(generate_vehicle_latents)
        2    0.051    0.025    0.051    0.025 {built-in method posix.read}



PASSED                                                                                        [100%]

========================================= warnings summary ==========================================
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
  /home/fireofearth/.local/miniconda3/envs/trajectron-cplex/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
================================== 1 passed, 8 warnings in 23.22s ===================================
```

```
(trajectron-cplex) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
=========================================================================================== test session starts ============================================================================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                                                                                                                           

tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000] 
Loading from /home/fireofearth/code/robotics/carla-collect/Trajectron-plus-plus/experiments/nuScenes/models/20210816/models_17_Aug_2021_13_25_38_carla_v3-1-2_base_distmapV4_modfm_K15_ph8/model_registrar-20.pt
Loaded!


---------------------------------------------------------------------------------------------- live log call -----------------------------------------------------------------------------------------------
INFO     numexpr.utils:utils.py:145 Note: NumExpr detected 12 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
INFO     numexpr.utils:utils.py:157 NumExpr defaulting to 8 threads.
Version identifier: 12.10.0.0 | 2019-11-26 | 843d4de
CPXPARAM_Read_DataCheck                          1
Tried aggregator 1 time.
Reduced MIQP has 176 rows, 80 columns, and 1344 nonzeros.
Reduced MIQP has 64 binaries, 0 generals, 0 SOSs, and 0 indicators.
Reduced MIQP objective Q matrix has 128 nonzeros.
Presolve time = 0.00 sec. (0.38 ticks)
Probing time = 0.00 sec. (0.02 ticks)
Tried aggregator 1 time.
Reduced MIQP has 176 rows, 80 columns, and 1344 nonzeros.
Reduced MIQP has 64 binaries, 0 generals, 0 SOSs, and 0 indicators.
Reduced MIQP objective Q matrix has 128 nonzeros.
Presolve time = 0.00 sec. (0.37 ticks)
Classifier predicts products in MIQP should be linearized.
Probing time = 0.00 sec. (0.02 ticks)
Clique table members: 32.
MIP emphasis: balance optimality and feasibility.
MIP search method: dynamic search.
Parallel mode: deterministic, using up to 12 threads.

Warning:  Barrier optimality criterion affected by large objective shift.
Root relaxation solution time = 0.01 sec. (2.80 ticks)

        Nodes                                         Cuts/
   Node  Left     Objective  IInf  Best Integer    Best Bound    ItCnt     Gap

      0     0       -0.0000    32                     -0.0000       11         
*     0+    0                         1526.5818       -0.0000           100.00%
*     0+    0                         1524.9823       -0.0000           100.00%
      0     0  -1.00000e+75     0     1524.9823       -0.0000       11  100.00%
      0     2       -0.0000    32     1524.9823       -0.0000       11  100.00%
Elapsed time = 0.05 sec. (29.85 ticks, tree = 0.02 MB, solutions = 2)

Root node processing (before b&c):
  Real time             =    0.05 sec. (29.61 ticks)
Parallel b&c, 12 threads:
  Real time             =    0.46 sec. (11.13 ticks)
  Sync time (average)   =    0.44 sec.
  Wait time (average)   =    0.00 sec.
                          ------------
Total (root+branch&cut) =    0.51 sec. (40.74 ticks)
INFO     root:util.py:67 code profile of __compute_prediction_controls
INFO     root:util.py:68          180453 function calls (170199 primitive calls) in 1.212 seconds

   Ordered by: cumulative time
   List reduced from 2654 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.212    1.212 __init__.py:511(__compute_prediction_controls)
        1    0.001    0.001    0.807    0.807 __init__.py:436(do_highlevel_control)
        1    0.000    0.000    0.513    0.513 model.py:4671(solve)
        1    0.000    0.000    0.511    0.511 model.py:4749(_solve_local)
        1    0.000    0.000    0.507    0.507 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.507    0.507 __init__.py:1282(solve)
        1    0.000    0.000    0.507    0.507 _procedural.py:689(mipopt)
        1    0.000    0.000    0.507    0.507 _pycplex.py:1532(CPXXmipopt)
        1    0.506    0.506    0.507    0.507 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
        1    0.000    0.000    0.398    0.398 __init__.py:226(do_prediction)
        1    0.000    0.000    0.349    0.349 scene.py:437(get_scene)
        1    0.001    0.001    0.225    0.225 trajectron_scene.py:454(process_scene)
        1    0.000    0.000    0.209    0.209 model.py:354(__init__)
        1    0.096    0.096    0.204    0.204 trajectron_scene.py:351(__process_carla_scene)
        2    0.000    0.000    0.186    0.093 environment.py:60(__init__)
        1    0.000    0.000    0.144    0.144 model.py:176(_make_environment)
        1    0.000    0.000    0.144    0.144 environment.py:426(get_default_env)
        1    0.000    0.000    0.144    0.144 environment.py:421(make_new_configured_env)
        1    0.000    0.000    0.124    0.124 scene.py:402(__checkpoint)
        1    0.035    0.035    0.103    0.103 scene.py:309(__process_lidar_snapshots)
     42/4    0.000    0.000    0.101    0.025 <frozen importlib._bootstrap_external>:672(exec_module)
     97/5    0.000    0.000    0.100    0.020 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
     45/7    0.000    0.000    0.098    0.014 <frozen importlib._bootstrap>:966(_find_and_load)
     45/7    0.000    0.000    0.098    0.014 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
     43/6    0.000    0.000    0.096    0.016 <frozen importlib._bootstrap>:651(_load_unlocked)
     43/4    0.000    0.000    0.095    0.024 {built-in method builtins.exec}
        2    0.000    0.000    0.095    0.048 environment.py:216(auto_configure)
        2    0.000    0.000    0.095    0.048 environment.py:323(check_cplex)
        3    0.000    0.000    0.095    0.032 environment.py:227(get_cplex_module)
        1    0.000    0.000    0.095    0.095 environment.py:276(load_cplex_from_cos_root)
        1    0.000    0.000    0.094    0.094 environment.py:249(load_cplex)
        2    0.000    0.000    0.090    0.045 platform.py:834(architecture)
        2    0.000    0.000    0.090    0.045 platform.py:798(_syscmd_file)
        1    0.000    0.000    0.090    0.090 __init__.py:33(<module>)
  512/438    0.000    0.000    0.088    0.000 <frozen importlib._bootstrap>:997(_handle_fromlist)
    21/16    0.000    0.000    0.088    0.005 {built-in method builtins.__import__}
        2    0.000    0.000    0.086    0.043 subprocess.py:608(__init__)
        2    0.000    0.000    0.086    0.043 subprocess.py:1228(_execute_child)
       42    0.000    0.000    0.085    0.002 <frozen importlib._bootstrap_external>:743(get_code)
        1    0.000    0.000    0.082    0.082 aborter.py:12(<module>)
        1    0.000    0.000    0.081    0.081 __init__.py:15(<module>)
       30    0.000    0.000    0.077    0.003 <frozen importlib._bootstrap_external>:735(source_to_code)
       30    0.077    0.003    0.077    0.003 {built-in method builtins.compile}
        1    0.000    0.000    0.060    0.060 model.py:342(_new_engine)
        1    0.000    0.000    0.060    0.060 model.py:1180(_make_new_engine_from_agent)
        1    0.000    0.000    0.060    0.060 engine_factory.py:90(new_engine)
        1    0.000    0.000    0.060    0.060 cplex_engine.py:463(__init__)
        1    0.000    0.000    0.060    0.060 cplex_adapter.py:49(__init__)
      320    0.000    0.000    0.050    0.000 linear.py:1013(times)
        1    0.000    0.000    0.049    0.049 __init__.py:348(compute_velocity_constraints)



PASSED                                                                                                                                                                                               [100%]

============================================================================================= warnings summary =============================================================================================
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
  /home/fireofearth/.local/miniconda3/envs/trajectron-cplex/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
====================================================================================== 1 passed, 8 warnings in 23.91s ======================================================================================
```
