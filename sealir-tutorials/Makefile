# Usage
# -----
#
# Initialize the notebooks with `make all`.
# Any update to the `.py` or `*.ipynb` files can be 
# synchronized by `make sync`. Synchronization works both ways, 
# edits on .ipynb will reflect in paired .py, and vice-versa.

# Define source directory and output directory
PY_FILES := $(wildcard *.py)
IPYNB_FILES := $(patsubst %.py,%.ipynb,$(PY_FILES))

# Default target to build all notebooks
all: $(IPYNB_FILES)

# Synchronize paired py-ipynb files
sync:
	jupytext --sync *.ipynb

# Run auto formatter
format:
	isort -l79 --profile black .
	black -l79 .

# Pattern rule to convert .py files to .ipynb
%.ipynb: %.py
	jupytext --update --to ipynb --from py:light $< -o $@

# Clean target to remove generated notebooks
clean:
	rm -rf *.ipynb

.PHONE: all sync dump