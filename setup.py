#!/usr/bin/env python3
"""
Setup script for REPLAY Influence Analysis
==========================================

This script enables proper installation of the REPLAY Influence Analysis package
with all dependencies and development tools.

Usage:
    pip install -e .                    # Development install
    pip install .                       # Regular install
    python setup.py sdist bdist_wheel   # Build distributions
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from src/__init__.py
def get_version():
    """Extract version from src/__init__.py"""
    init_file = Path(__file__).parent / "src" / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding='utf-8')
        match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
        if match:
            return match.group(1)
    return "0.1.0"  # Fallback version

# Read long description from README
def get_long_description():
    """Get long description from README.md"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding='utf-8')
    return "REPLAY Influence Analysis - State-of-the-art influence function analysis for deep learning models."

# Core dependencies
install_requires = [
    "torch>=2.2.2,<3.0.0",
    "torchvision>=0.17.2,<1.0.0", 
    "numpy>=1.26.4,<2.0.0",
    "matplotlib>=3.10.3,<4.0.0",
    "seaborn>=0.13.2,<1.0.0",
    "scipy>=1.15.3,<2.0.0",
    "tqdm>=4.67.1,<5.0.0",
]

# Development dependencies  
dev_requires = [
    "pytest>=8.0.0,<9.0.0",
    "pytest-cov>=4.0.0,<5.0.0",
    "black>=24.0.0,<25.0.0",
    "mypy>=1.8.0,<2.0.0",
    "flake8>=7.0.0,<8.0.0",
    "isort>=5.13.0,<6.0.0",
    "pre-commit>=3.6.0,<4.0.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=7.0.0,<8.0.0",
    "sphinx-rtd-theme>=2.0.0,<3.0.0",
    "myst-parser>=2.0.0,<3.0.0",
]

setup(
    name="replay-influence-analysis",
    version=get_version(),
    author="Yingru Li",
    author_email="szrlee@gmail.com",
    description="State-of-the-art influence function analysis for deep learning models",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    # url="https://github.com/replay-influence/replay-influence-analysis",
    
    packages=find_packages(),
    package_data={
        "src": ["*.py"],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "all": dev_requires + docs_requires,
    },
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "replay-influence=main_runner:main",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for discovery
    keywords=[
        "influence-functions",
        "machine-learning", 
        "deep-learning",
        "pytorch",
        "interpretability",
        "explainable-ai",
        "data-valuation",
        "model-debugging",
    ],
    
    ## Project URLs
    # project_urls={
    #     "Documentation": "https://replay-influence.readthedocs.io/",
    #     "Source": "https://github.com/replay-influence/replay-influence-analysis",
    #     "Tracker": "https://github.com/replay-influence/replay-influence-analysis/issues",
    # },
    
    # Include package data files
    include_package_data=True,
    
    # Zip safety
    zip_safe=False,
    
    # Minimum setuptools version
    setup_requires=["setuptools>=45", "wheel"],
) 