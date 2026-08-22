import os
import sys
from setuptools import Extension, setup

extra_compile_args = ["-O3"]
if sys.platform != "win32":
    extra_compile_args.extend(["-fPIC", "-Wall"])

fast_ops_module = Extension(
    "quran_forced_align._fast_ops",
    sources=["src/quran_forced_align/_fast_ops.c"],
    extra_compile_args=extra_compile_args,
    include_dirs=["src/quran_forced_align"],
)

setup(
    ext_modules=[fast_ops_module],
)
