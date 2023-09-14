#!/bin/bash

# write pre-commit config options
/bin/cat << END_CONFIG > .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
        args: [--force-exclude, "maillib.py"]
END_CONFIG

pre-commit install
