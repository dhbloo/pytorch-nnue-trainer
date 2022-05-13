from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name='line_encoding_cpp',
    ext_modules=[
        Pybind11Extension(name='line_encoding_cpp',
                          sources=['line_encoding.cpp']),
    ],
    cmdclass={"build_ext": build_ext},
)