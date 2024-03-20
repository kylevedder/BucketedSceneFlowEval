#!/bin/bash

echo "Running integration tests"
pytest tests/integration_tests.py tests/eval/*.py tests/datasets/*/*.py tests/datastructures/*.py
