import os
from setuptools import setup, find_packages

def make_directories(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
required_packages = [l.strip() for l in open('requirements.txt', 'r').readlines() if l != '\n']

make_directories("data")
make_directories("models")
make_directories("out_data")
make_directories("out_data/sensitivities")
make_directories("out_data/sensitivities/cases")
make_directories("out_data/sensitivities/controls")

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

