
Code is slow:

```
(trajectron-cplex) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
======================================== test session starts ========================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                    

tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn] 
Loading from /home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/models/20210622/models_19_Mar_2021_22_14_19_int_ee_me_ph8/model_registrar-20.pt
Loaded!


------------------------------------------- live log call -------------------------------------------
INFO     root:__init__.py:103 code profile of __compute_prediction_controls
INFO     root:__init__.py:104          206668 function calls (196555 primitive calls) in 8.594 seconds

   Ordered by: cumulative time
   List reduced from 2424 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    8.594    8.594 __init__.py:554(__compute_prediction_controls)
        1    0.002    0.002    8.115    8.115 __init__.py:273(do_prediction)
        1    0.000    0.000    8.055    8.055 scene.py:424(get_scene)
        1    0.002    0.002    7.728    7.728 scene.py:388(__checkpoint)
      381    0.414    0.001    7.673    0.020 scene.py:309(__process_lidar_snapshot)
     1625    7.187    0.004    7.187    0.004 {built-in method numpy.concatenate}
        1    0.003    0.003    0.477    0.477 __init__.py:454(do_highlevel_control)
        1    0.003    0.003    0.327    0.327 trajectron_scene.py:533(process_scene)
        1    0.156    0.156    0.302    0.302 trajectron_scene.py:430(__process_carla_scene)
        1    0.000    0.000    0.254    0.254 model.py:4671(solve)
        1    0.000    0.000    0.254    0.254 model.py:4749(_solve_local)
        1    0.000    0.000    0.252    0.252 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.252    0.252 __init__.py:1282(solve)
        1    0.000    0.000    0.252    0.252 _procedural.py:689(mipopt)
        1    0.000    0.000    0.252    0.252 _pycplex.py:1532(CPXXmipopt)
        1    0.251    0.251    0.252    0.252 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
        1    0.000    0.000    0.144    0.144 model.py:354(__init__)
        2    0.000    0.000    0.117    0.059 environment.py:60(__init__)
        1    0.000    0.000    0.106    0.106 model.py:176(_make_environment)
        1    0.000    0.000    0.106    0.106 environment.py:426(get_default_env)
        1    0.000    0.000    0.106    0.106 environment.py:421(make_new_configured_env)
     36/4    0.000    0.000    0.097    0.024 <frozen importlib._bootstrap_external>:672(exec_module)
     88/5    0.000    0.000    0.095    0.019 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        2    0.000    0.000    0.094    0.047 environment.py:216(auto_configure)
        2    0.000    0.000    0.094    0.047 environment.py:323(check_cplex)
        3    0.000    0.000    0.094    0.031 environment.py:227(get_cplex_module)
        1    0.000    0.000    0.094    0.094 environment.py:276(load_cplex_from_cos_root)
        1    0.000    0.000    0.094    0.094 environment.py:249(load_cplex)
     39/7    0.000    0.000    0.093    0.013 <frozen importlib._bootstrap>:966(_find_and_load)
     39/7    0.000    0.000    0.093    0.013 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)



PASSED                                                                                        [100%]

========================================= warnings summary ==========================================
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
tests/test_in_simulation_v2.py::test_Town03_scenario[ovehicle_turn]
  /home/fireofearth/.local/miniconda3/envs/trajectron-cplex/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
================================== 1 passed, 8 warnings in 25.82s ===================================
```

```
(trajectron-cplex) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
======================================== test session starts ========================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                    

tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000] 
Loading from /home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/models/20210622/models_19_Mar_2021_22_14_19_int_ee_me_ph8/model_registrar-20.pt
Loaded!


------------------------------------------- live log call -------------------------------------------
INFO     root:__init__.py:103 code profile of __compute_prediction_controls
INFO     root:__init__.py:104          279614 function calls (269522 primitive calls) in 9.277 seconds

   Ordered by: cumulative time
   List reduced from 2411 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    9.277    9.277 __init__.py:556(__compute_prediction_controls)
        1    0.002    0.002    7.932    7.932 __init__.py:273(do_prediction)
        1    0.000    0.000    7.880    7.880 scene.py:424(get_scene)
        1    0.002    0.002    7.555    7.555 scene.py:388(__checkpoint)
      386    0.385    0.001    7.495    0.019 scene.py:309(__process_lidar_snapshot)
     1686    7.037    0.004    7.037    0.004 {built-in method numpy.concatenate}
        1    0.030    0.030    1.338    1.338 __init__.py:456(do_highlevel_control)
    20000    0.732    0.000    0.833    0.000 util.py:23(get_vertices_from_center)
        1    0.003    0.003    0.325    0.325 trajectron_scene.py:534(process_scene)
        1    0.155    0.155    0.304    0.304 trajectron_scene.py:431(__process_carla_scene)
        1    0.000    0.000    0.266    0.266 model.py:4671(solve)
        1    0.000    0.000    0.265    0.265 model.py:4749(_solve_local)
        1    0.000    0.000    0.264    0.264 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.263    0.263 __init__.py:1282(solve)
        1    0.000    0.000    0.263    0.263 _procedural.py:689(mipopt)
        1    0.000    0.000    0.262    0.262 _pycplex.py:1532(CPXXmipopt)
        1    0.262    0.262    0.262    0.262 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
        1    0.000    0.000    0.190    0.190 model.py:354(__init__)
        2    0.000    0.000    0.161    0.080 environment.py:60(__init__)
        1    0.000    0.000    0.128    0.128 model.py:176(_make_environment)
        1    0.000    0.000    0.128    0.128 environment.py:426(get_default_env)
        1    0.000    0.000    0.128    0.128 environment.py:421(make_new_configured_env)
84592/84572    0.127    0.000    0.128    0.000 {built-in method numpy.array}
     36/4    0.000    0.000    0.095    0.024 <frozen importlib._bootstrap_external>:672(exec_module)
        1    0.000    0.000    0.095    0.095 scene.py:476(points_to_2d_histogram)
        1    0.001    0.001    0.095    0.095 twodim_base.py:571(histogram2d)
     88/5    0.000    0.000    0.094    0.019 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        1    0.002    0.002    0.093    0.093 histograms.py:924(histogramdd)
        2    0.000    0.000    0.093    0.046 environment.py:216(auto_configure)
        2    0.000    0.000    0.093    0.046 environment.py:323(check_cplex)



PASSED                                                                                        [100%]

========================================= warnings summary ==========================================
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
tests/test_closed_loop.py::test_Town03_scenario[ovehicle_turn_short-ph8_ch8_np5000]
  /home/fireofearth/.local/miniconda3/envs/trajectron-cplex/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
================================== 1 passed, 8 warnings in 27.29s ===================================
```
