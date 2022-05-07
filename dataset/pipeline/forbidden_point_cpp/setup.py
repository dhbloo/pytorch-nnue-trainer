from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name='forbidden_point_cpp',
    ext_modules=[
        Pybind11Extension(name='forbidden_point_cpp',
                          sources=[
                              'forbidden_point.cpp',
                              'board.cpp',
                          ]),
    ],
    cmdclass={"build_ext": build_ext},
)