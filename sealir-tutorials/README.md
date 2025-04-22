# Tutorial

This is kaleidoscope-style tutorial that systematically explains and
demonstrates key concepts for Numba v2, and progressively building an advanced
compiler that integrates Egraph and MLIR.

## Code Structure

- Each chapter has prefix `chNN_`, where `NN` is the chapter number.
- Each chapter is written as a runnable script and in convertible into notebook via jupytext.
- Each chapter is a Python module that can optionally depends on earlier chapters.
    - Any executable code for demonstrating must be guarded inside a `if __name__ == "__main__"` so the file can be imported cleanly without side-effects
- Each chapter has accompanying tests in the `./tests/test_chNN.py`


## Setup

```bash
conda env create -f=./conda_environment.yml -n sealir_tutorial
conda activate sealir_tutorial
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

## Rendering notebooks to HTML

Run `make pages` to render the notebooks to HTML for export to GitHub pages.
Output will be in the `../pages/sealir-tutorials` subdirectory.  Remember to keep
the Markdown table of contents in `index.py` file up-to-date when new notebooks
are added.

You can view pre-rendered notebooks at: https://numba.pydata.org/numba-prototypes/sealir_tutorials/
