
Code is slow:

```
(trajectron) fireofearth@daedalus:~/code/robotics/carla-collect$ pytest --log-cli-level=INFO --capture=tee-sys tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000]
======================================== test session starts ========================================
platform linux -- Python 3.6.12, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
benchmark: 3.2.3 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/fireofearth/code/robotics/carla-collect
plugins: benchmark-3.2.3
collected 1 item                                                                                    

tests/test_closed_loop.py::test_Town06_scenario[merge_lane_short-ph8_ch8_np5000] 
Loading from /home/fireofearth/code/robotics/trajectron-plus-plus/experiments/nuScenes/models/20210622/models_19_Mar_2021_22_14_19_int_ee_me_ph8/model_registrar-20.pt
Loaded!


------------------------------------------- live log call -------------------------------------------
INFO     root:__init__.py:104 code profile of __compute_prediction_controls
INFO     root:__init__.py:105          178518 function calls (168644 primitive calls) in 1.314 seconds

   Ordered by: cumulative time
   List reduced from 2419 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    1.314    1.314 __init__.py:564(__compute_prediction_controls)
        1    0.002    0.002    0.885    0.885 __init__.py:491(do_highlevel_control)
        1    0.000    0.000    0.632    0.632 model.py:4671(solve)
        1    0.000    0.000    0.632    0.632 model.py:4749(_solve_local)
        1    0.000    0.000    0.630    0.630 cplex_engine.py:1898(solve)
        1    0.000    0.000    0.629    0.629 __init__.py:1282(solve)
        1    0.000    0.000    0.629    0.629 _procedural.py:689(mipopt)
        1    0.000    0.000    0.629    0.629 _pycplex.py:1532(CPXXmipopt)
        1    0.629    0.629    0.629    0.629 {built-in method cplex._internal.py36_cplex12100.CPXXmipopt}
        1    0.000    0.000    0.427    0.427 __init__.py:274(do_prediction)
        1    0.000    0.000    0.366    0.366 scene.py:468(get_scene)
        1    0.001    0.001    0.238    0.238 trajectron_scene.py:534(process_scene)
        1    0.102    0.102    0.216    0.216 trajectron_scene.py:431(__process_carla_scene)
        1    0.000    0.000    0.212    0.212 model.py:354(__init__)
        2    0.000    0.000    0.186    0.093 environment.py:60(__init__)
        1    0.000    0.000    0.143    0.143 model.py:176(_make_environment)
        1    0.000    0.000    0.143    0.143 environment.py:426(get_default_env)
        1    0.000    0.000    0.143    0.143 environment.py:421(make_new_configured_env)
        1    0.000    0.000    0.128    0.128 scene.py:434(__checkpoint)
        1    0.037    0.037    0.108    0.108 scene.py:309(__process_lidar_snapshots)
     36/4    0.000    0.000    0.102    0.025 <frozen importlib._bootstrap_external>:672(exec_module)
     88/5    0.000    0.000    0.100    0.020 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        2    0.000    0.000    0.099    0.050 environment.py:216(auto_configure)
        2    0.000    0.000    0.099    0.050 environment.py:323(check_cplex)
        3    0.000    0.000    0.099    0.033 environment.py:227(get_cplex_module)
     39/7    0.000    0.000    0.098    0.014 <frozen importlib._bootstrap>:966(_find_and_load)
        1    0.000    0.000    0.098    0.098 environment.py:276(load_cplex_from_cos_root)
        1    0.000    0.000    0.098    0.098 environment.py:249(load_cplex)
     39/7    0.000    0.000    0.098    0.014 <frozen importlib._bootstrap>:936(_find_and_load_unlocked)
     36/6    0.000    0.000    0.096    0.016 <frozen importlib._bootstrap>:651(_load_unlocked)
     37/4    0.000    0.000    0.096    0.024 {built-in method builtins.exec}
        1    0.000    0.000    0.093    0.093 __init__.py:33(<module>)
  499/425    0.000    0.000    0.091    0.000 <frozen importlib._bootstrap>:997(_handle_fromlist)
    20/15    0.000    0.000    0.091    0.006 {built-in method builtins.__import__}
       36    0.000    0.000    0.087    0.002 <frozen importlib._bootstrap_external>:743(get_code)
        2    0.000    0.000    0.087    0.043 platform.py:834(architecture)
        2    0.000    0.000    0.086    0.043 platform.py:798(_syscmd_file)
        1    0.000    0.000    0.085    0.085 aborter.py:12(<module>)
        1    0.000    0.000    0.084    0.084 __init__.py:15(<module>)
        2    0.000    0.000    0.082    0.041 subprocess.py:608(__init__)
        2    0.000    0.000    0.082    0.041 subprocess.py:1228(_execute_child)
       30    0.000    0.000    0.080    0.003 <frozen importlib._bootstrap_external>:735(source_to_code)
       30    0.080    0.003    0.080    0.003 {built-in method builtins.compile}
        1    0.000    0.000    0.065    0.065 model.py:342(_new_engine)
        1    0.000    0.000    0.065    0.065 model.py:1180(_make_new_engine_from_agent)
        1    0.000    0.000    0.064    0.064 engine_factory.py:90(new_engine)
        1    0.000    0.000    0.064    0.064 cplex_engine.py:463(__init__)
        1    0.000    0.000    0.064    0.064 cplex_adapter.py:49(__init__)
        1    0.000    0.000    0.061    0.061 prediction.py:19(generate_vehicle_latents)
        2    0.049    0.024    0.049    0.024 {built-in method posix.read}



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
  /home/fireofearth/.local/miniconda3/envs/trajectron/lib/python3.6/site-packages/control/statesp.py:103: PendingDeprecationWarning: the matrix subclass is not the recommended way to represent matrices or deal with linear algebra (see https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). Please adjust your code to use regular ndarray.
    arr = np.matrix(data, dtype=float)

-- Docs: https://docs.pytest.org/en/stable/warnings.html
================================== 1 passed, 8 warnings in 22.43s ===================================
```
