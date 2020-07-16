
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
from wrapper_ewald import Sfac, ewald_insertion, ewald_from_sfac
from wrapper_forceparts import electrostatics, electrostatics_realspace_insert
from molmod.units import kelvin, bar, kjmol, angstrom
from yaff import log
log.set_level(0)


# Test the reciprocal ewald energy of the insertion of 2 particles
def test_ewald_insert():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/CO2.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    rcut = 15 * angstrom
    fugacity = P

    # Try 10 configurations
    for i in range(10):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)
        sfac = Sfac(mcmd.pos, mcmd.N_frame, mcmd.rvecs_flat, mcmd.charges, mcmd.alpha, mcmd.gcut)

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        new_pos2 = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos2 = np.append(pos, new_pos2, axis=0)

        Z_ads = 2
        plen = len(pos)
        plen2 = len(pos2)

        sfac_insert_1 = np.array(ewald_insertion(sfac, new_pos, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        sfac_full_1 = Sfac(pos, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.charges[:plen], mcmd.alpha, mcmd.gcut)
        assert (np.abs(sfac_full_1 - sfac_insert_1) < 1e-8).all()
        e_insert_1 = ewald_from_sfac(sfac_insert_1, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)
        e_full_1 = ewald_from_sfac(sfac_full_1, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)
        assert (np.abs(e_full_1 - e_insert_1) < 1e-8).all()


        sfac_insert_2 = np.array(ewald_insertion(sfac_insert_1, new_pos2, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        sfac_full_2 = Sfac(pos2, mcmd.N_frame, mcmd.rvecs_flat, mcmd.data.charges[:plen2], mcmd.alpha, mcmd.gcut)
        assert (np.abs(sfac_full_2 - sfac_insert_2) < 1e-8).all()
        e_insert_2 = ewald_from_sfac(sfac_insert_2, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)
        e_full_2 = ewald_from_sfac(sfac_full_2, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)
        assert (np.abs(e_full_2 - e_insert_2) < 1e-8).all()
        print(e_full_2/kjmol, e_insert_2/kjmol)


# Test the realspace ei energy of the insertion of 2 particles
def test_ei_insert():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/CO2.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    rcut = 15 * angstrom
    fugacity = P

    # Try 10 configurations
    for i in range(10):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)
        sfac = Sfac(mcmd.pos, mcmd.N_frame, mcmd.rvecs_flat, mcmd.charges, mcmd.alpha, mcmd.gcut)

        N_frame = len(mcmd.pos)
        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos = np.append(mcmd.pos, new_pos, axis=0)

        new_pos2 = random_ads(mcmd.pos_ads, mcmd.rvecs)
        pos2 = np.append(pos, new_pos2, axis=0)

        Z_ads = 2
        plen = len(pos)
        plen2 = len(pos2)

        ei_insert_1 = electrostatics_realspace_insert(mcmd.N_frame, len(pos)-mcmd.nads, pos, mcmd.rvecs_flat, mcmd.data.charges[:plen], mcmd.data.radii[:plen], mcmd.rcut, mcmd.alpha, mcmd.gcut)
        sfac = np.array(ewald_insertion(sfac, new_pos, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        ew_insert_1 = ewald_from_sfac(sfac, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)

        ei_full_1 = electrostatics(pos, mcmd.N_frame, 1, mcmd.rvecs_flat, mcmd.data.charges[:plen], mcmd.data.radii[:plen], mcmd.rcut, mcmd.alpha, mcmd.gcut)

        print(ei_insert_1/kjmol, ew_insert_1/kjmol, ei_full_1/kjmol)
        assert (np.abs(ei_full_1 - ei_insert_1 - ew_insert_1) < 1e-8).all()


        ei_insert_2 = ei_insert_1 + electrostatics_realspace_insert(mcmd.N_frame, len(pos2)-mcmd.nads, pos2, mcmd.rvecs_flat, mcmd.data.charges[:plen2], mcmd.data.radii[:plen2], mcmd.rcut, mcmd.alpha, mcmd.gcut)
        sfac = np.array(ewald_insertion(sfac, new_pos2, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        ew_insert_2 = ewald_from_sfac(sfac, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)

        ei_full_2 = electrostatics(pos2, mcmd.N_frame, 2, mcmd.rvecs_flat, mcmd.data.charges[:plen2], mcmd.data.radii[:plen2], mcmd.rcut, mcmd.alpha, mcmd.gcut)

        print(ei_insert_2/kjmol, ew_insert_2/kjmol, ei_full_2/kjmol)
        assert (np.abs(ei_full_2 - ei_insert_2 - ew_insert_2) < 1e-8).all()






# Test the rcut dependence of the ei energy
def test_ei_rcut():
    system_file = pkg_resources.resource_filename(__name__, '../data/lp_avg.chk')
    ff_file = pkg_resources.resource_filename(__name__, '../data/pars.txt')
    adsorbate_file = pkg_resources.resource_filename(__name__, '../data/CO2.chk')

    T = 87 * kelvin
    P = 1 * bar
    MD_trial_fraction = 0.000
    fugacity = P


    def get_total_ei(rcut, new_pos, new_pos2):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = 25)
        sfac = Sfac(mcmd.pos, mcmd.N_frame, mcmd.rvecs_flat, mcmd.charges, mcmd.alpha, mcmd.gcut)

        pos = np.append(mcmd.pos, new_pos, axis=0)
        pos2 = np.append(pos, new_pos2, axis=0)
        Z_ads = 2
        plen = len(pos)
        plen2 = len(pos2)

        ei_insert_1 = electrostatics_realspace_insert(mcmd.N_frame, len(pos)-mcmd.nads, pos, mcmd.rvecs_flat, mcmd.data.charges[:plen], mcmd.data.radii[:plen], rcut, mcmd.alpha, mcmd.gcut)
        sfac = np.array(ewald_insertion(sfac, new_pos, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        ew_insert_1 = ewald_from_sfac(sfac, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)

        ei_insert_2 = ei_insert_1 + electrostatics_realspace_insert(mcmd.N_frame, len(pos2)-mcmd.nads, pos2, mcmd.rvecs_flat, mcmd.data.charges[:plen2], mcmd.data.radii[:plen2], rcut, mcmd.alpha, mcmd.gcut)
        sfac = np.array(ewald_insertion(sfac, new_pos2, mcmd.rvecs_flat, mcmd.data.charges_ads, mcmd.alpha, mcmd.gcut))
        ew_insert_2 = ewald_from_sfac(sfac, mcmd.rvecs_flat, mcmd.alpha, mcmd.gcut)

        ei_full_2 = electrostatics(pos2, mcmd.N_frame, 2, mcmd.rvecs_flat, mcmd.data.charges[:plen2], mcmd.data.radii[:plen2], rcut, mcmd.alpha, mcmd.gcut)

        return ew_insert_2 + ei_insert_2, ei_full_2


    # Try 10 configurations
    for i in range(10):

        mcmd = MCMD(system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, 30*angstrom, fixed_N = 25)
        sfac = Sfac(mcmd.pos, mcmd.N_frame, mcmd.rvecs_flat, mcmd.charges, mcmd.alpha, mcmd.gcut)

        new_pos = random_ads(mcmd.pos_ads, mcmd.rvecs)
        new_pos2 = random_ads(mcmd.pos_ads, mcmd.rvecs)

        ei_base, ei_base_check = get_total_ei(30*angstrom, new_pos, new_pos2)
        print('base: ', ei_base/kjmol)
        assert(np.abs(ei_base - ei_base_check) < 1e-8)

        for rcut in 25, 20, 15, 12, 10:
            ei, ei_check = get_total_ei(rcut*angstrom, new_pos, new_pos2)
            print(rcut, ei/kjmol)
            assert np.abs(ei_base - ei)/kjmol < 0.2
            assert(np.abs(ei - ei_check) < 1e-8)
