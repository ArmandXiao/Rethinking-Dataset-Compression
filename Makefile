.PHONY: build install reinstall clean refresh

# Command to build the package
build:
	python -m build

# Command to install the package in editable mode
install:
	pip install -e .

# Command to uninstall and reinstall the package in editable mode
reinstall:
	pip uninstall -y rethinkdc && pip install -e .

# Command to clean up build artifacts
clean:
	rm -rf build/ dist/ *.egg-info

# Command to clean, build, and reinstall the package
refresh: clean
	$(MAKE) install
