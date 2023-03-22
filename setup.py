#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.build_scripts import build_scripts


class CMakeExt(Extension):
    def __init__(self, name, cmdir="."):
        Extension.__init__(self, name, [])
        self.cmake_lists_dir = os.path.abspath(cmdir)


from distutils.dep_util import newer
from distutils import log
from stat import ST_MODE
from distutils.util import convert_path


class BinBuild(build_scripts):
    def initialize_options(self):
        build_scripts.initialize_options(self)
        self.build_temp = None

    def finalize_options(self):
        build_scripts.finalize_options(self)
        self.set_undefined_options("build", ("build_temp", "build_temp"))

    def copy_scripts(self):
        self.scripts = [x for x in self.scripts if x != "block2"]
        outfiles, updated_files = build_scripts.copy_scripts(self)
        self.scripts += ["block2"]
        for script in ["block2"]:
            script = os.path.join(self.build_temp, script)
            script = convert_path(script)
            outfile = os.path.join(self.build_dir, os.path.basename(script))
            outfiles.append(outfile)

            if not self.force and not newer(script, outfile):
                log.debug("not copying %s (up-to-date)", script)
                continue

            updated_files.append(outfile)
            self.copy_file(script, outfile)

        if os.name == "posix":
            for file in outfiles:
                if self.dry_run:
                    log.info("changing mode of %s", file)
                else:
                    oldmode = os.stat(file)[ST_MODE] & 0o7777
                    newmode = (oldmode | 0o555) & 0o7777
                    if newmode != oldmode:
                        log.info(
                            "changing mode of %s from %o to %o", file, oldmode, newmode
                        )
                        os.chmod(file, newmode)
        return outfiles, updated_files


class CMakeBuild(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(["cmake", "--version"])
            print(out.decode("utf-8"))
        except OSError:
            raise RuntimeError("Cannot find CMake executable!")

        print("Python3: ", sys.executable)
        print("Build Dir: ", self.build_temp)

        for ext in self.extensions:

            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cfg = "Release"

            cmake_args = [
                "-DCMAKE_BUILD_TYPE=%s" % cfg,
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(
                    cfg.upper(), self.build_temp
                ),
                "-DPYTHON_EXECUTABLE_HINT={}".format(sys.executable),
                "-DUSE_MKL=ON",
                "-DBUILD_LIB=ON",
                "-DLARGE_BOND=ON",
                "-DUSE_KSYMM=OFF",
                "-DUSE_COMPLEX=ON",
                "-DUSE_SINGLE_PREC=ON",
                "-DUSE_SG=ON",
            ]

            # We can handle some platform-specific settings at our discretion
            if platform.system() == "Windows":
                plat = "x64" if platform.architecture()[0] == "64bit" else "Win32"
                cmake_args += [
                    "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
                    "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(
                        cfg.upper(), extdir
                    ),
                ]
                if self.compiler.compiler_type == "msvc":
                    cmake_args += [
                        "-DCMAKE_GENERATOR_PLATFORM=%s" % plat,
                    ]
                else:
                    cmake_args += [
                        "-G",
                        "MinGW Makefiles",
                    ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            subprocess.check_call(
                ["cmake", "--build", ".", "--config", cfg, "--", "--jobs=2"],
                cwd=self.build_temp,
            )

            cmake_args = [x for x in cmake_args if x != "-DBUILD_LIB=ON"]
            cmake_args += ["-DBUILD_LIB=OFF"]

            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            subprocess.check_call(
                ["cmake", "--build", ".", "--config", cfg, "--", "--jobs=2"],
                cwd=self.build_temp,
            )


from distutils.command.build import build

build.sub_commands = [c for c in build.sub_commands if c[0] == "build_ext"] + [
    c for c in build.sub_commands if c[0] != "build_ext"
]


setup(
    name="block2",
    version="0.1.10",
    packages=find_packages(),
    ext_modules=[CMakeExt("block2")],
    cmdclass={"build_ext": CMakeBuild, "build_scripts": BinBuild},
    license="LICENSE",
    description="""An efficient MPO implementation of DMRG for quantum chemistry.""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Huanchen Zhai, Henrik R. Larsson, Seunghoon Lee, and Zhi-Hao Cui",
    author_email="hczhai.ok@gmail.com",
    url="https://github.com/block-hczhai/block2-preview",
    install_requires=[
        "mkl==2021.4",
        "mkl-include",
        "intel-openmp",
        "numpy",
        "cmake>=3.19",
        "scipy",
        "psutil",
        "pybind11<=2.10.1",
    ],
    scripts=[
        "pyblock2/driver/block2main",
        "pyblock2/driver/gaopt",
        "pyblock2/driver/readwfn.py",
        "pyblock2/driver/writewfn.py",
        "block2",
    ],
)

