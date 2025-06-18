#!/bin/bash
set -euo pipefail

# Install the package using pip
python -m pip install . -vv --no-deps --no-build-isolation

# Copy tutorial files to a shared location for easy access
mkdir -p $PREFIX/share/sealir-tutorials
cp -r *.py $PREFIX/share/sealir-tutorials/
cp -r *.ipynb $PREFIX/share/sealir-tutorials/
cp -r tests $PREFIX/share/sealir-tutorials/
cp Makefile $PREFIX/share/sealir-tutorials/
cp README.md $PREFIX/share/sealir-tutorials/
cp jupytext.toml $PREFIX/share/sealir-tutorials/
cp pyproject.toml $PREFIX/share/sealir-tutorials/