# Quantum Entangled MaxCut Solver (QEMC)

A quantum computing implementation for solving the Maximum Cut (MaxCut) problem using PennyLane and Qiskit, with support for both quantum simulators and real quantum hardware (IBM Quantum).

## Paper Reference

This code implements the algorithms described in:

**"A quantum entanglement-based approach for computing Maximum Cut in graphs"**  
arXiv:2308.10383  
Link: https://arxiv.org/abs/2308.10383

## Overview

This project implements a quantum algorithm for solving the MaxCut problem, which is a fundamental NP-hard optimization problem in graph theory. The implementation uses variational quantum algorithms with entangled quantum circuits to find approximate solutions.

### Key Features

- Quantum circuit implementation using PennyLane
- Support for both noiseless simulation and real quantum hardware (IBM Quantum)
- Database integration for experiment tracking and result storage
- Comparison with classical Goemans-Williamson algorithm
- Scalable to graphs with up to 2048+ nodes
- Comprehensive visualization and analysis tools

## Project Structure

```
├── Database Management
│   ├── DbAdapter.py              # Database interface for experiment tracking
│   └── DbAdapterGraph.py         # Graph-specific database operations
│
├── Quantum Algorithms
│   ├── with_ent_general_db.py    # Main quantum algorithm with entanglement
│   ├── no_ent_general.py         # Non-entangled version
│   └── with_ent_circular.py      # Circular graph specialization
│
├── Classical Algorithms
│   ├── Goemans-Williamson.py     # Classical GW algorithm implementation
│   └── max_cut_exhaustive_search.py  # Exhaustive search for small graphs
│
├── Visualization & Analysis
│   ├── DrawGraphFromDb.py        # Database result visualization
│   ├── plot.py, plot2.py, plot3.py  # Various plotting utilities
│   └── show_graph.py             # Graph structure visualization
│
├── Experiment Runners
│   ├── HWRuns.py                 # Hardware experiment orchestration
│   ├── ShotsSymRuns.py           # Shot noise analysis
│   └── runner_*.py               # Various experiment runners
│
└── Utilities
    ├── create_graph.py           # Graph generation utilities
    ├── create_graph_regular.py   # Regular graph generation
    └── convert_to_g6.py          # Graph format conversion
```

## Requirements

### Python Dependencies

```bash
pip install pennylane
pip install qiskit
pip install qiskit-ibmq-provider
pip install networkx
pip install numpy
pip install matplotlib
pip install cvxpy
pip install mariadb
pip install pandas
pip install tqdm
pip install numba
pip install natsort
```

### Database Setup

The project uses MariaDB for experiment tracking. Set up a database named `maxcut` with the following tables:

- `experiments` - Main experiment metadata
- `iterations` - Per-iteration results
- `loss` - Loss function values
- `maxcut` - MaxCut values found
- `maxcut_max` - Maximum maxcut values
- `maxcut_max_1000` - Maximum maxcut values for 1000 iterations
- `groups` - Node groupings
- `probs` - Probability distributions
- `blacks` - Black node counts

Database connection configuration (in DbAdapter.py):
```python
user="root"
password="root"
host="localhost"
port=3306
database="maxcut"
```

## Usage

### 1. Generate Graphs

Create regular graphs for experiments:
```bash
python create_graph_regular.py
```

Create general graphs:
```bash
python create_graph.py <count> <size>
```

Example:
```bash
python create_graph.py 10 5  # Creates 10 graphs with 2^5 = 32 nodes
```

### 2. Run Quantum Experiments

#### Simulator Experiments
```bash
python with_ent_general_db.py <expected_blacks> <graph_file> <layers> <stepsize> <steps> <shots> <backend> <comment> <expected_maxcut>
```

Example:
```bash
python with_ent_general_db.py 8 graphs/vertices_16.g6 2 0.7 350 768 sym "test_run" 21
```

Parameters:
- `expected_blacks`: Target number of nodes in one partition (usually n/2 for n nodes)
- `graph_file`: Path to graph file in g6 format
- `layers`: Number of entangling layers in the quantum circuit
- `stepsize`: Learning rate for the optimizer
- `steps`: Number of optimization iterations
- `shots`: Number of measurement shots (use -1 for exact simulation)
- `backend`: Either 'sym' for simulator or IBM backend name
- `comment`: Description for the experiment
- `expected_maxcut`: Known optimal maxcut value (for early stopping)

#### Hardware Experiments (IBM Quantum)
```bash
python HWRuns.py
```

### 3. Classical Comparison

Run Goemans-Williamson algorithm:
```bash
python Goemans-Williamson.py <graph_file> <num_trials>
```

Example:
```bash
python Goemans-Williamson.py graphs/vertices_16.g6 10
```

### 4. Analyze Results

Generate visualizations from database:
```bash
python DrawGraphFromDb.py
```

Plot experiment results:
```bash
python plot.py
```

## IBM Quantum Access

To use IBM Quantum hardware:

1. Get an IBM Quantum account at https://quantum-computing.ibm.com/
2. Get your API token from your IBM Quantum dashboard
3. Replace the token in the code with your credentials:

```python
provider = IBMQ.enable_account(
    "YOUR_TOKEN_HERE",
    hub='YOUR_HUB', 
    group='YOUR_GROUP', 
    project='YOUR_PROJECT'
)
```

## Graph File Format

The project uses NetworkX's graph6 format (.g6 files) for graph storage. You can convert edge lists to g6 format using:
```bash
python convert_to_g6.py
```

## Experiment Metadata

The following metadata is used for different graph sizes:

| Nodes | Stepsize | Layers | Mean GW | Max GW |
|-------|----------|--------|---------|---------|
| 8     | 0.8      | 1      | 9.2     | 10      |
| 16    | 0.7      | 5      | 45.5    | 46      |
| 32    | 0.7      | 5      | 98.3    | 102     |
| 64    | 0.1      | 50     | 199.5   | 204     |
| 128   | 0.2      | 40     | 407.4   | 415     |
| 256   | 0.08     | 80     | 817.0   | 830     |
| 512   | 0.1      | 70     | 1633.8  | 1672    |
| 1024  | 0.12     | 100    | 3285.8  | 3307    |
| 2048  | 0.08     | 120    | 6580.3  | 6630    |

## Results

The implementation achieves:
- Near-optimal solutions for small graphs (up to 16 nodes)
- Competitive performance compared to Goemans-Williamson for medium graphs
- Scalability demonstrations up to 2048+ nodes
- Hardware validation on IBM Quantum devices

## Citation

If you use this code in your research, please cite:

```bibtex
@article{qemc2023,
  title={A quantum entanglement-based approach for computing Maximum Cut in graphs},
  author={[Authors]},
  journal={arXiv preprint arXiv:2308.10383},
  year={2023}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- IBM Quantum Network for hardware access
- PennyLane team for quantum computing framework
- NetworkX for graph processing capabilities
- The authors of the Goemans-Williamson algorithm
