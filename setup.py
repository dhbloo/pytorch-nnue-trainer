"""Builds the pybind11 C++ extensions; all other metadata lives in pyproject.toml."""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    ext_modules=[
        Pybind11Extension(
            "line_encoding_cpp",
            sources=["dataset/pipeline/line_encoding_cpp/line_encoding.cpp"],
        ),
        Pybind11Extension(
            "forbidden_point_cpp",
            sources=[
                "dataset/pipeline/forbidden_point_cpp/forbidden_point.cpp",
                "dataset/pipeline/forbidden_point_cpp/board.cpp",
            ],
        ),
        Pybind11Extension(
            "dataset_planner_cpp",
            sources=["dataset/planner_cpp/packed_shuffle.cpp"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
