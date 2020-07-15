"""
Unit and regression test for the osmotic_mcmd package.
"""

# Import package, test suite, and other packages as needed
import osmotic_mcmd
import pytest
import sys
import pkg_resources


#from osmotic_mcmd.tests.common import get_system_cof300, get_ff_cof300, get_system_argon
from osmotic_mcmd.mcmd import MCMD
from molmod.units import kelvin, bar, kjmol, angstrom

def test_osmotic_mcmd_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "osmotic_mcmd" in sys.modules


def test_mcmd_system():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    rcut = 15 * angstrom

    from yaff.pes.eos import PREOS
    eos = PREOS.from_name('argon')
    fugacity = eos.calculate_fugacity(T,P)
    mu = eos.calculate_mu(T,P)
    print('Mu: ', mu/kjmol)
    print('Fug: ', fugacity/bar)

    mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)
    mcmd.run_GCMC(20000, 1000)



