import pytest

def pytest_configure(config):
    config.option.benchmark_columns = ['mean', 'stddev']