# PAC One-Step Safety Certification for Black-Box Discrete-Time Stochastic Systems

This repository provides a prototype tool that implements the framework proposed in the paper *PAC One-Step Safety Certification for Black-Box Discrete-Time Stochastic Systems*. Specifically, it offers a data-driven framework for one-step safety certification of black-box discrete-time stochastic systems, where both the system dynamics and disturbance distributions are unknown and only sampled data are available. Building on robust and stochastic barrier certificates, the repository implements three methods for establishing formal one-step safety guarantees based on scenario approaches, VC dimension, and Markov’s and Hoeffding’s inequalities.  This one-step guarantee can be applied recursively at each time step, thereby yielding step-by-step safety assurances over extended horizons. In addition, we provide a set of benchmark examples.



## Installation

1. Install [Gurobi](https://www.gurobi.com/downloads/), noting that a license is required. An academic license can be obtained by applying [here](https://www.gurobi.com/features/academic-named-user-license/). The version of Gurobi Optimizer used in this paper is 11.0.3.



2. Run the following commands to create an environment and install the necessary dependencies. Python version 3.9 is recommended. 

```bash
conda create --name conda_name python=3.9

conda activate conda_name

python pip install -e .
```



## How to use

The `benchmarks` folder contains three subdirectories — `RBC-I`, `RBC-II`, and `SBC-III` — corresponding to the methods introduced in Theorems 2, 3, and 4 of the paper. Running the Python scripts in these subdirectories will reproduce the associated results. For example, to run the example in the `RBC-I` folder, execute:

```bash
python ex1.py
```



The tool is designed to be easily extensible to new systems. Additional system models can be added in the `src/model/stochastic` directory.



# Case 2 Video in the CARLA Simulator

The video below demonstrates the application of our method in the CARLA simulator, corresponding to Case 2: Car Following in the paper:

https://github.com/user-attachments/assets/de418eda-95cd-4e8c-86cd-0a085d482d98

