# SNNexperiments
Reinforcement learning framework for spiking neural networks actors with R-STDP for the master thesis "Training Spiking Neural Networks with Reinforcement Learning".

## Requirements
Python 3.7 or 3.8

## Installation
First, nest 3 needs to be installed.

### Compile NEST 3 on macOS:

From nest dir, replace :
```
cmake -DCMAKE_INSTALL_PREFIX:PATH=~/Studium/Masterarbeit/nest \
-DCMAKE_C_COMPILER=/usr/local/bin/gcc-10\
-DCMAKE_CXX_COMPILER=/usr/local/bin/g++-10 \
./
```

then `make install` and afterwards `source ~/.bashrc`

### Install NEST via Docker
Alternatively install via the docker image by running nestdockerrun.sh or usin this line:
```docker run --rm -it -e LOCAL_USER_ID=`id -u $USER` -v $(pwd):/opt/data -p 8080:8080 nestsim/nest:latest /bin/bash```

then attach with

```docker ps```

```docker attach```

## Install Python Dependencies
run

```pip install -r requirements.txt```


## Running Experiments
Default experiment

`python exp.py`

Some predefined experiments are included in the experiments directory

Grid search on global values can be performed by adding the argument

```-g <gridsearchparameter.json>```

to any experiment.

Example files are in the `gridsearch` directory. The parameter names must match the parameters in the globalvalues.py.

## Obtaining Results
Plots are written to ./experimentdata/

If a mongoDB connection string is specified in the exp.py, writing to the db can be enabled. Then training progress and data dumps are written to the db. 

## Config
`render` enables live rendering

`headless` should be enabled when running on a server without a display driver
