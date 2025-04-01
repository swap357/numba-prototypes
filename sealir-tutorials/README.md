# Tutorial


## Setup

```bash
conda env create -f=./conda_environment.yml -n sealir_tutorial
conda activate sealir_tutorial
pip install git+https://github.com/numba/sealir
```

## Working on the notebooks

Run `make all` to initialize `notebooks/*.ipynb` files.

It can be easier to edit the markdown in `.ipynb` via visual editors.
Edits can be synchronized to the paired py-ipynb files by `make sync`


## Ch 1. Basic Compiler

Showcase a basic SealIR compiler without a middle-end. Only frontend and backend.

### What it does?

- Compiles a function-at-a-time into LLVM
- LLVM uses Python C-API
- Assumes to run as a JIT


### Run

```bash
python ch01_basic_compiler.py
```


## Ch 2. EGraph Basic

Showcase EGraph roundtripping in the middle-end.


### What it does?

- Adds middle-end to Ch01
- Adds roundtripping from RVSDG to EGraph and back to RVSDG

### Run

```bash
python ch02_egraph_basic.py
```
