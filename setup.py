import os
from setuptools import setup, find_packages

required_packages = [l.strip() for l in open('requirements.txt', 'r').readlines() if l != '\n']
setup(
    name = "dyslexialearn",
    version = "0.0.1",
    author = "Foram Joshi",
    author_email = "foram2494@gmail.com",
    description = ("Train and Explain Dyslexia from 3D brain MRI scan using Deep Learning"),
    packages=find_packages(),
    install_requires=required_packages,
    include_package_data=True
)

