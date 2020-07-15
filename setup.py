from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "osmotic_mcmd"
VERSION = "0.1"
DESCR = "A package to perform mcmd simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Ruben Goeminne"
EMAIL = "Ruben.Goeminne@UGent.be"

LICENSE = "GPLv3"

SRC_DIR = "osmotic_mcmd"
PACKAGES = [SRC_DIR]

ext_ew = Extension("wrapper_ewald",
                  [SRC_DIR + "/wrapper_ewald.pyx", SRC_DIR + "/cell.c",
                   SRC_DIR + "/ewald.c"],
                  libraries=[],
                  include_dirs=[np.get_include()])

ext_force = Extension("wrapper_forceparts",
                  [SRC_DIR + "/wrapper_forceparts.pyx", SRC_DIR + "/cell.c",
                   SRC_DIR + "/ewald.c", SRC_DIR + "/forceparts.c"],
                  libraries=[],
                  include_dirs=[np.get_include()])



EXTENSIONS = [ext_ew, ext_force]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )








"""
osmotic_mcmd
A package to perform mcmd simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine

import sys
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import build_ext
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

SRC_DIR = 'osmotic_mcmd'

setup(
    # Self-descriptive entries which should always be present
    name='osmotic_mcmd',
    author='Ruben Goeminne',
    author_email='Ruben.Goeminne@UGent.be',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    license='LGPLv3',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    cmdclass = {'build_ext': build_ext},

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

    ext_modules = [
        Extension(SRC_DIR + "/cext",
            sources = [SRC_DIR + "/cext.pyx", SRC_DIR + "/forceparts.c",
                       SRC_DIR + "/cell.c", SRC_DIR + "/ewald.c"],
            include_dirs=[np.get_include()],
            language='c',
            extra_compile_args=['-O3']
        )
    ]


)

"""
