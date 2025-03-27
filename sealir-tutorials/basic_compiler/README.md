# Basic Compiler

Showcase a basic SealIR compiler without a middle-end. Only frontend and backend.

## What it does?

- Compiles a function-at-a-time into LLVM
- LLVM uses Python C-API
- Assumes to run as a JIT


## Setup

```bash
conda env create -f=./conda_environment.yml -n sealir_basic_compiler
conda activate sealir_basic_compiler
pip install git+https://github.com/numba/sealir
```

## Run

```bash
python main.py
```
