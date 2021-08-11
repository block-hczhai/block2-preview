#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExt(Extension):
    def __init__(self, name, cmdir='.'):
        Extension.__init__(self, name, [])
        self.cmake_lists_dir = os.path.abspath(cmdir)


class CMakeBuild(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(['cmake', '--version'])
            print(out.decode("utf-8"))
        except OSError:
            raise RuntimeError('Cannot find CMake executable!')

        print('Python3: ', sys.executable)
        print('Build Dir: ', self.build_temp)

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(
                self.get_ext_fullpath(ext.name)))
            cfg = 'Release'

            cmake_args = [
                '-DCMAKE_BUILD_TYPE=%s' % cfg,
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir),
                '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), self.build_temp),
                '-DPYTHON_EXECUTABLE_HINT={}'.format(sys.executable),
                '-DUSE_MKL=ON',
                '-DBUILD_LIB=ON',
                '-DLARGE_BOND=ON'
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == 'Windows':
                plat = ('x64' if platform.architecture()
                        [0] == '64bit' else 'Win32')
                cmake_args += [
                    '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE',
                    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(
                        cfg.upper(), extdir)
                ]
                if self.compiler.compiler_type == 'msvc':
                    cmake_args += [
                        '-DCMAKE_GENERATOR_PLATFORM=%s' % plat,
                    ]
                else:
                    cmake_args += [
                        '-G', 'MinGW Makefiles',
                    ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(['cmake', ext.cmake_lists_dir] + cmake_args,
                                  cwd=self.build_temp)

            subprocess.check_call(['cmake', '--build', '.', '--config', cfg, '--', '--jobs=4'],
                                  cwd=self.build_temp)


setup(name='block2',
      version='0.1.10',
      packages=find_packages(),
      ext_modules=[CMakeExt('block2')],
      cmdclass={'build_ext': CMakeBuild},
      license='LICENSE',
      description="""An efficient highly scalable implementation of the
        DMRG for quantum chemistry, based on MPO formalism.""",
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Huanchen Zhai, Henrik R. Larsson, and Zhi-Hao Cui',
      author_email='hczhai@ucla.edu',
      url='https://github.com/block-hczhai/block2-preview',
      install_requires=[
          "mkl",
          "mkl-include",
          "intel-openmp",
          "numpy",
          "cmake==3.17",
          "scipy",
          "psutil",
          "pybind11"
      ],
      scripts=["pyblock2/driver/block2main", "pyblock2/driver/gaopt",
               "pyblock2/driver/readwfn.py", "pyblock2/driver/writewfn.py"]
      )
