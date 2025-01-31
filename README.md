# Dynamic Multi-Swarm Particle Swarm Optimization (DMS-PSO)

## Overview

This repository contains the implementation of multiple Particle Swarm Optimization (PSO) variants, including the Dynamic Multi-Swarm PSO (DMS-PSO). The project is based on the research paper *"Dynamic Multi-Swarm Particle Swarm Optimizer"* by J.J. Liang and P.N. Suganthan.

### **Implemented PSO Variants:**

- `cpso` - Cooperative PSO
- `fdr_pso` - Fitness-Distance-Ratio PSO
- `pso_cf` - PSO with Constriction Factor (Global)
- `pso_cf_local` - PSO with Constriction Factor (Local)
- `pso_w` - PSO with Inertia Weight (Global)
- `pso_w_local` - PSO with Inertia Weight (Local)
- `upso` - Unified PSO
- `dms_pso` - Dynamic Multi-Swarm PSO (Proposed method)

## Features

- Implementation of **eight different PSO algorithms**.
- Performance evaluation using **six benchmark functions**:
  - Sphere function
  - Rosenbrock’s function
  - Ackley’s function
  - Griewank’s function
  - Rastrigin’s function
  - Weierstrass function
- Comparison of results between different PSO variants.
- Demonstrates the **advantages of DMS-PSO** over standard PSO algorithms.

## Installation

To run the implementations, ensure you have Python installed along with the following dependencies:

```sh
pip install numpy matplotlib
```

## Usage

Run any of the PSO implementations using:

```sh
python dms_pso.py
```

To compare all PSO variants:

```sh
python benchmark_test.py
```

## Results

The results demonstrate that **DMS-PSO performs better on multimodal functions** and adapts dynamically to the optimization problem by **regrouping and enhancing diversity**.

### **Screenshots of Results**
A `results` folder is included in the repository, containing **eight subfolders** (one for each PSO variant). Each subfolder contains **screenshots of output results** showcasing the performance of each algorithm on different benchmark functions.

## Contributions

Contributors:

- **G. Ashrith**
- **P. Veeresh Kumar**
- **Rishika Reddy**
- **Tarun Kumar Revelli**

Feel free to contribute, improve, or extend this repository by opening issues or pull requests!

## License

This project is released under the **MIT License**.

