"""Shared pytest configuration.

This replication uses no market data: every experiment is a numerical
verification of closed-form theorems or a Monte Carlo simulation of models
fully specified by the paper (rating chains, Cox hazards, CIR intensities).
The ``market_data`` marker is registered for consistency with the project's
testing conventions even though no test in this suite currently needs it.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test depends on a live market-data API call"
    )
