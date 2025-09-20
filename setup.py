#!/usr/bin/env python3
"""
Setup script for Wearable Data Analysis Package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wearable-data-analysis",
    version="1.0.0",
    author="Wearable Data Analysis Team",
    author_email="contact@example.com",
    description="A comprehensive machine learning pipeline for analyzing wearable sensor data",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fcookie/wearable-data-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.15.0",
            "notebook>=6.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "wearable-analysis=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    project_urls={
        "Bug Reports": "https://github.com/fcookie/wearable-data-analysis/issues",
        "Source": "https://github.com/fcookie/wearable-data-analysis",
        "Documentation": "https://github.com/fcookie/wearable-data-analysis/wiki",
    },
)