# Displaced_GBS_Backend

This Github repository provides a calculation backend for Displaced Gaussian Boson Sampling. It provides methods to 
convert between graph representation, gaussian state representation and experiment circuit representation of 
Displaced Gaussian Boson Sampling (D-GBS), as well as scripts to compute output photon number distribution, benchmark 
complexity transition of D-GBS, numerically simulate the distribution of zeroes of the loop-Hafnian, and 
parallel batch-compute loop-Hafnians to analyse their distribution. They may provide a useful complement to 
common quantum optics libraries such as [Strawberry Fields](https://strawberryfields.ai/) and 
[The Walrus](https://the-walrus.readthedocs.io/). 

The methods and conventions are based on our paper, [A complexity transition in displaced Gaussian Boson sampling](https://doi.org/10.1038/s41534-025-01062-5). 
Plotting scripts for the figures in the paper are also provided within this repository. Citations where appropriate would be appreciated. 

## Installation 
Users interested in using the code are welcome to fork this repository. 

## Required Packages 
The code is written for Python 3.9 and requires the following packages:
- `numpy == 1.26.3`
- `scipy == 1.11.4`
- `matplotlib == 3.5.1`
- `pandas == 1.3.5`
- `sympy == 1.9`
- `tqdm=4.65.0`
- `networkx == 2.6.3`
- `mpmath==1.2.1`
- `numba == 0.59.0`
- `strawberryfields == 0.23.0`
- `thewalrus == 0.19.0`

An easy way to install the required packages is to create a conda environment from the environment2.yml file, which 
specifies an anaconda environment. In your anaconda prompt or command line, type the following to create the 
environment, `GBS`:
```bash
conda env create -f environment_dgbs.yml
```
Then activate the environment:
```bash
conda activate GBS
```

## Citing 
If you use this code in your research, please cite [our paper](https://www.nature.com/articles/s41534-025-01062-5) 
where appropriate: 

Zhenghao Li, Naomi R. Solomons, Jacob F.F. Bulmer, Raj B. Patel, and Ian A. Walmsley, 
A complexity transition in displaced Gaussian Boson sampling. _npj Quantum Inf_ **11**, 119 (2025). 
https://doi.org/10.1038/s41534-025-01062-5

```bibtex
@article{Li_DGBS_2025, 
    title={A complexity transition in displaced Gaussian Boson sampling}, 
    volume={11}, 
    url={https://www.nature.com/articles/s41534-025-01062-5}, 
    DOI={https://doi.org/10.1038/s41534-025-01062-5}, 
    number={1}, 
    journal={npj Quantum Information}, 
    publisher={Nature Portfolio}, 
    author={Li, Zhenghao and Solomons, Naomi R and Jacob and Patel, Raj B and Walmsley, Ian A}, 
    year={2025}, 
    month={Jul} 
 }
```