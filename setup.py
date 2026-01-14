"""
Setup script for aircraft-control-algorithms.

The C++ extension is optional - a pre-built .so file is included for convenience.
To rebuild from source, set BUILD_CPP_EXTENSION=1:
    BUILD_CPP_EXTENSION=1 pip install -e .
"""

from setuptools import setup, find_packages
import os

__version__ = "0.1.0"

# Check if we should attempt to build C++ extensions
BUILD_CPP = os.environ.get("BUILD_CPP_EXTENSION", "0") == "1"

ext_modules = []
cmdclass = {}

if BUILD_CPP:
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext

        # C++ source files (actual paths)
        cpp_sources = [
            "cpp/src/pid_controller.cpp",
            "cpp/bindings/bindings.cpp",
        ]

        # Only build if source files exist
        if all(os.path.exists(src) for src in cpp_sources):
            ext_modules = [
                Pybind11Extension(
                    "aircraft_controls_bindings",
                    cpp_sources,
                    include_dirs=["cpp/include"],
                    cxx_std=17,
                    define_macros=[("VERSION_INFO", __version__)],
                ),
            ]
            cmdclass = {"build_ext": build_ext}
            print("C++ extension will be built from source.")
        else:
            missing = [src for src in cpp_sources if not os.path.exists(src)]
            print(f"Warning: C++ source files not found: {missing}")
            print("Skipping C++ extension build. Using pre-built .so if available.")
    except ImportError:
        print("Warning: pybind11 not available. Skipping C++ extension build.")
        print("The pre-built aircraft_controls_bindings.*.so should still work.")

# Read README for long description
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="aircraft-control-algorithms",
    version=__version__,
    author="Lucas Wright",
    author_email="lhwright13@users.noreply.github.com",
    description="Aircraft control algorithms demonstration with hybrid Python/C++ implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lhwright13/controls",
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
    cmdclass=cmdclass,
    zip_safe=False,
    package_data={
        "": ["*.so", "*.pyd", "*.dylib"],
    },
    include_package_data=True,
)
