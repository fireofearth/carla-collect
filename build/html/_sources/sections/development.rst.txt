Development
===========

Technical Goals
---------------

- Datasets for Trajectron++ should NOT be saved in `.pkl` as Python objects (i.e. using `dill`)
  since the codebase and Python library version becomes coupled with the dataset,
  and it is not possible to load the dataset if the codebase changes in certain but common cases.
  Instead use something like [h5py](https://docs.h5py.org/en/stable/quick.html) more for storing numerical data.
- Ideally, use CARLA 0.9.13, but it requies 8 GB GPU memory. 
  CARLA 0.9.11 works as an alternative, but it's not possible to fix autopilot car routes using this version.
- Ideally, both `carla-collect` and Trajectron++ should be upgraded to Torch 1.11 so it is compatible with newer graphics cards.

Lessons
-------

- Trajectron++ was initially released for Python 3.6 but the latest releases of PyTorch and up-to-date libraries are only compatible with >=3.7.
- CPLEX 20.1.0 does not provide Matlab support, but still provides Python support.
- CPLEX 20.1.0/docplex is only compatible with Python >=3.8.
-  
