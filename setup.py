from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

__version__ = "0.1.0"

# C++ extension modules
ext_modules = [
    Pybind11Extension(
        "controls_core",
        [
            "core/pid_controller.cpp",
            "core/control_mixer.cpp",
            "core/attitude_control.cpp",
            "core/bindings.cpp",
        ],
        include_dirs=["core"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aircraft-control-algorithms",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    description="Aircraft control algorithms demonstration with hybrid Python/C++ implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/controls",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "pandas>=1.3.0",
        "h5py>=3.0.0",
        "dash>=2.0.0",
        "dash-bootstrap-components>=1.0.0",
        "pybind11>=2.9.0",
        "dataclasses-json>=0.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "pyvista>=0.35.0",
            "kaleido>=0.2.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
