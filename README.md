
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
(trajectron-cplex) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
======================================== test session starts ========================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                    

tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane] 
Loading from /home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/models/20210622/models_19_Mar_2021_22_14_19_int_ee_me_ph8/model_registrar-20.pt
Loaded!


------------------------------------------- live log call -------------------------------------------
INFO     root:__init__.py:104 code profile of do_highlevel_control
INFO     root:__init__.py:105          327037 function calls (326068 primitive calls) in 2.245 seconds

   Ordered by: cumulative time
   List reduced from 1397 to 30 due to restriction <30>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001    2.245    2.245 __init__.py:487(do_highlevel_control)
        1    0.055    0.055    1.672    1.672 __init__.py:455(__compute_vertices)
    40000    1.416    0.000    1.617    0.000 util.py:23(get_vertices_from_center)
        1    0.000    0.000    0.401    0.401 model.py:4671(solve)
        1    0.000    0.000    0.401    0.401 model.py:4749(_solve_local)
        1    0.000    0.000    0.399    0.399 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.398    0.398 __init__.py:1282(solve)
        1    0.000    0.000    0.398    0.398 _procedural.py:689(mipopt)
        1    0.000    0.000    0.398    0.398 _pycplex.py:1532(CPXXmipopt)
        1    0.398    0.398    0.398    0.398 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
   160158    0.191    0.000    0.191    0.000 {built-in method numpy.array}
        1    0.000    0.000    0.137    0.137 model.py:354(__init__)
        2    0.000    0.000    0.112    0.056 environment.py:60(__init__)
        1    0.000    0.000    0.102    0.102 model.py:176(_make_environment)
        1    0.000    0.000    0.102    0.102 environment.py:426(get_default_env)
        1    0.000    0.000    0.102    0.102 environment.py:421(make_new_configured_env)
     34/3    0.000    0.000    0.093    0.031 <frozen importlib._bootstrap_external>:672(exec_module)
     86/4    0.000    0.000    0.091    0.023 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        2    0.000    0.000    0.091    0.045 environment.py:216(auto_configure)
        2    0.000    0.000    0.091    0.045 environment.py:323(check_cplex)
        3    0.000    0.000    0.091    0.030 environment.py:227(get_cplex_module)
        1    0.000    0.000    0.090    0.090 environment.py:276(load_cplex_from_cos_root)
        1    0.000    0.000    0.090    0.090 environment.py:249(load_cplex)
     36/6    0.000    0.000    0.089    0.015 <frozen importlib._bootstrap>:966(_find_and_load)
     36/6    0.000    0.000    0.089    0.015 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
     34/5    0.000    0.000    0.087    0.017 <frozen importlib._bootstrap>:651(_load_unlocked)
     35/3    0.000    0.000    0.087    0.029 {built-in method builtins.exec}
        1    0.000    0.000    0.085    0.085 __init__.py:33(<module>)
   161/87    0.000    0.000    0.083    0.001 <frozen importlib._bootstrap>:997(_handle_fromlist)
    20/15    0.000    0.000    0.083    0.006 {built-in method builtins.__import__}



PASSED                                                                                        [100%]

========================================= warnings summary ==========================================
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
tests/test_in_simulation_v2.py::test_Town06_scenario[merge_lane]
  /home/fireofearth/.local/miniconda3/envs/trajectron-cplex/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
================================== 1 passed, 8 warnings in 20.15s ===================================
```
