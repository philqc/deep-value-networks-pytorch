#!/bin/sh

flake8 || exit 1
mypy --ignore-missing-imports . || exit 1