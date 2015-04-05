#!/usr/bin/env python
import pkg_resources
import sys
from distutils.core import setup
sys.path.insert(0, '.')
import version

from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np
from path_helpers import path


pyx_files = ['cyplace_experiments/data/CONNECTIONS_TABLE.pyx']
ext_modules = [Extension(f[:-4].replace('/', '.'), [f],
                         extra_compile_args=['-O3', '-msse3', '-std=c++0x'],
                         include_dirs=[path('~/local/include').expand(),
                                       '/usr/local/cuda-6.5/include',
                                       pkg_resources
                                       .resource_filename('cythrust', ''),
                                       np.get_include()],
                         define_macros=[('THRUST_DEVICE_SYSTEM',
                                         'THRUST_DEVICE_SYSTEM_CPP')])
               for f in pyx_files]


setup(name="fpga_netlist",
      version=version.getVersion(),
      description="FPGA netlist models and utilities.",
      keywords="fpga netlist python",
      author="Christian Fobel <christian@fobel.net>",
      url="https://github.com/cfobel/fpga_netlist",
      license="GPL",
      long_description="""""",
      packages=['fpga_netlist'],
      ext_modules=cythonize(ext_modules))
