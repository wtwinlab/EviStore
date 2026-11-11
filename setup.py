#!/usr/bin/env python3
"""Installable package definition for DTRUST."""

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dtrust",
    version="1.0.0",
    description="DTRUST: evidence-driven dynamic trust management for federated learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    packages=find_packages(exclude=("experiments", "experiments.*")),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        # scripts under experiments/
        "experiments": ["matplotlib>=3.7", "duckdb>=0.9", "pyarrow>=14.0", "numba>=0.58"],
    },
    include_package_data=True,
    entry_points={"console_scripts": ["dtrust=dtrust.cli:cli"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
