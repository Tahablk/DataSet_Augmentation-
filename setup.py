from setuptools import setup, find_packages
from os import path
from codecs import open

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="perturbationdrive",
    version="0.0.1",
    description="A library to test the robustness of Self-Driving Cars",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://",
    author="Hannes Leonhard",
    author_email="example@email.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "scipy",
        "matplotlib",
        "scikit-image",
        "requests",
        "Pillow",
        "tensorflow",  # Use this for compatibility with windows
        "tensorflow-addons",
    ],
    python_requires=">=3",
)
