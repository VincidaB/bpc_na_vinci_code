# How to test the code

All of the tests run everytime something is pushed to the repository, or when a pull request is opened. This is done using GitHub Actions.

## Installation

To install the code, you can use the following command from the root of the repo (preferably in a virtual environment) :

```bash
pip install -e .
```

## Testing

This repo uses:
- `pytest` and `pytest-cov` for general testing (very quick, use as much as possible)
- `tox` for testing on multiple python versions (useful for CI, may be long to run locally)
- `flake8` for linting
- `mypy` for static typing

To run the tests with `pytest`, you can use the following command from the root of the repo:

```bash
pytest
```

To run all the tests, you can use the following command from the root of the repo:

```bash
tox
```

## Adding tests

New tests should be added in the `our_tests/` directory. The tests should be named `test_<name>.py` where `<name>` is the name of the module you are testing.
The tests function should be named `test_<name_of_the_function>`. You can use the `pytest` framework to write your tests (see documentation [here](https://docs.pytest.org/en/stable/)).
