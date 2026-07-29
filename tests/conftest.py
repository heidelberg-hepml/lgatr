"""Shared pytest configuration."""

import zlib

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def seed(request: pytest.FixtureRequest) -> None:
    # Seed from the test id rather than a constant, so different parametrizations still see
    # different draws while every failure stays reproducible.
    value = zlib.crc32(request.node.nodeid.encode()) % 2**31
    torch.manual_seed(value)
    np.random.seed(value)
