"""
Unit and regression test for the osmotic_mcmd package.
"""

# Import package, test suite, and other packages as needed
import osmotic_mcmd
import pytest
import sys
import pkg_resources
from yaff import *
import h5py as h5

#from osmotic_mcmd.tests.common import get_system_cof300, get_ff_cof300, get_system_argon
from osmotic_mcmd.mcmd import Widom
from molmod.units import kelvin, bar, kjmol, angstrom, femtosecond



def test_widom_system():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 300 * kelvin
    rcut = 15 * angstrom

    widom = Widom(system_file, adsorbate_file, ff_file, T, rcut)
    widom.run_widom(25000, 5000)





