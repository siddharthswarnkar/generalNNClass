
import sys
from setuptools import setup, find_packages

requires = ["numpy"]

setup(
        name="neuralNetworkClass",
        version="1.0",
        description="General neural network class",
        url="https://github.com/siddharthswarnkar/generalNNClass",
        author="M Suriya Kumar, Siddharth Swarnkar",
        author_email="msuriyak.2495@gmail.com, siddharthswarnkar@gmail.com",
        install_requires=requires,
        packages=find_packages(),
        )
