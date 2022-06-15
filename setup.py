#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

readme = open('README.md').read()
VERSION = "0.1.1"

requirements = [
    "torch",
    "torchvision"
]

# import subprocess
# commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True).decode('UTF-8').rstrip()
# VERSION += "_" + str(int(commit_hash, 16))[:8]
VERSION += "_" + datetime.datetime.now().strftime("%Y%m%d%H%M")

setup(
    # Metadata
    name="mcunet",
    version=VERSION,
    author="MTI HAN LAB ",
    author_email="hanlab.eecs+github@gmail.com",
    url="https://github.com/mit-han-lab/mcunet",
    description="MCUNet: Tiny Deep Learning on IoT Devices",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    packages=find_packages(exclude=("*test*",)),
    #
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
