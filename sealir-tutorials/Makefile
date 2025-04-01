# Usage
# -----
#
# Initialize the notebooks with `make all`.
# Any update to the `.py` or `notebooks/*.ipynb` files can be 
# synchronized by `make sync`. Synchronization works both ways, 
# edits on .ipynb will reflect in paired .py, and vice-versa.

# Define source directory and output directory
PY_FILES := $(wildcard *.py)
IPYNB_FILES := $(patsubst %.py,notebooks/%.ipynb,$(PY_FILES))

# Default target to build all notebooks
all: $(IPYNB_FILES)

# Synchronize paired py-ipynb files
sync:
	jupytext --sync notebooks/*.ipynb

# Pattern rule to convert .py files to .ipynb
notebooks/%.ipynb: %.py
	@mkdir -p notebooks
	jupytext --update --to ipynb --from py:light $< -o $@

# Clean target to remove generated notebooks
clean:
	rm -rf notebooks/*.ipynb

.PHONE: all sync dump