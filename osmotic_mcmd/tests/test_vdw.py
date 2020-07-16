"""
Unit and regression test for the osmotic_mcmd package.
"""

# Import package, test suite, and other packages as needed
import osmotic_mcmd
import pytest
import sys
import pkg_resources
import numpy as np

from osmotic_mcmd.mcmd import MCMD
from osmotic_mcmd.utilities import Acceptance, Parse_data, random_ads, random_rot
from wrapper_forceparts import MM3, MM3_insert, LJ, LJ_insert
from molmod.units import kelvin, bar, kjmol, angstrom
from yaff import log
log.set_level(0)


# Test the mm3 energy of the insertion of 2 particles
def test_mm3_insert():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    rcut = 15 * angstrom
    fugacity = P

    # Try 10 configurations
    for i in range(10):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        new_pos2 = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos2 = np.append(pos, new_pos2, axis=0)

        Z_ads = 2
        plen = len(pos)
        plen2 = len(pos2)

        e_vdw = MM3(pos2, N_frame, Z_ads, mcmd.rvecs_flat, mcmd.data.sigmas[:plen2], mcmd.data.epsilons[:plen2], rcut)

        e_vdw_insert = MM3_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], mcmd.rcut)
        e_vdw_insert += MM3_insert(pos2, len(pos2)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen2], mcmd.data.epsilons[:plen2], mcmd.rcut)
        print(e_vdw/kjmol, e_vdw_insert/kjmol)

        assert np.abs(e_vdw - e_vdw_insert) < 1e-8


# Test the lj energy of the insertion of 2 particles
def test_lj_insert():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    rcut = 15 * angstrom
    fugacity = P

    # Try 10 configurations
    for i in range(10):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        new_pos2 = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos2 = np.append(pos, new_pos2, axis=0)

        Z_ads = 2
        plen = len(pos)
        plen2 = len(pos2)

        e_vdw = LJ(pos2, N_frame, Z_ads, mcmd.rvecs_flat, mcmd.data.sigmas[:plen2], mcmd.data.epsilons[:plen2], rcut)

        e_vdw_insert = LJ_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], mcmd.rcut)
        e_vdw_insert += LJ_insert(pos2, len(pos2)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen2], mcmd.data.epsilons[:plen2], mcmd.rcut)
        print(e_vdw/kjmol, e_vdw_insert/kjmol)

        assert np.abs(e_vdw - e_vdw_insert) < 1e-8



# Test the tail corrections of the mm3 potential
# An insertion energy shouldn't deviate more than ~0.2 kJ/mol for a cutoff of 10 angstrom, compared to 30 angstrom
def test_mm3_tail():

    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    fugacity = P

    mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, 8*angstrom, fixed_N = 25)

    # Try 10 configurations
    for i in range(10):

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        Z_ads = 1
        plen = len(pos)

        e_vdw_insert_base = MM3_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], 30*angstrom)
        print('base: ', e_vdw_insert_base/kjmol)

        for rcut in 25, 20, 15, 12, 10:
            e_vdw_insert = MM3_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], rcut*angstrom)
            print(rcut, e_vdw_insert/kjmol)
            assert np.abs(e_vdw_insert_base - e_vdw_insert)/kjmol < 0.2


# Test the tail corrections of the mm3 potential
# An insertion energy shouldn't deviate more than ~0.2 kJ/mol for a cutoff of 10 angstrom, compared to 30 angstrom
def test_lj_tail():

    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/argon.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    fugacity = P

    mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, 8*angstrom, fixed_N = 25)

    # Try 10 configurations
    for i in range(10):

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        Z_ads = 1
        plen = len(pos)

        e_vdw_insert_base = LJ_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], 30*angstrom)
        print('base: ', e_vdw_insert_base/kjmol)

        for rcut in 25, 20, 15, 12, 10:
            e_vdw_insert = LJ_insert(pos, len(pos)-mcmd.nads, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.sigmas[:plen], mcmd.data.epsilons[:plen], rcut*angstrom)
            print(rcut, e_vdw_insert/kjmol)
            assert np.abs(e_vdw_insert_base - e_vdw_insert)/kjmol < 0.2



