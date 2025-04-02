# Tutorial

## Setup

```bash
conda env create -f=./conda_environment.yml -n sealir_tutorial
conda activate sealir_tutorial
pip install git+https://github.com/numba/sealir
```

## Working on the notebooks

Run `make all` to initialize `*.ipynb` files.

It can be easier to edit the markdown in `.ipynb` via visual editors.
Edits can be synchronized to the paired py-ipynb files by `make sync`

Run `make format` to use `black` and `isort` to auto-format the scripts. 

To remove the `*.ipynb` files, run `make clean`.

## Testing

Tests are located in `./tests` directory.

Use `pytest` to run all tests.
