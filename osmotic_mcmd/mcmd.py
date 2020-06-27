"""
mcmd.py
A package to perform simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine

Handles the primary functions
"""

import numpy as np
from molmod.constants import boltzmann
from molmod.units import angstrom, kjmol, kelvin, femtosecond, bar, kcalmol

from osmotic_mcmd.utilities import Acceptance, Parse_data, random_ads, random_rot

class MCMD():
    def __init__(self, system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut):

        self.ff_file = ff_file
        self.T = T
        self.beta = 1/(boltzmann*T)
        self.P = P
        self.fugacity = fugacity
        self.prob = np.array([0.5, 0.5, MD_trial_fraction], dtype=float)
        self.prob = np.cumsum(self.prob)/sum(self.prob)
        self.rcut = rcut

        data = Parse_data(system_file, adsorbate_file, ff_file)
        self.data = data

        print('ei: ', data.ei)
        print('lj: ', data.lj)
        print('mm3: ', data.mm3)

        self.pos = data.pos_MOF
        self.N_frame = len(self.pos)
        self.charges = data.charges_MOF
        self.pos_ads = data.pos_ads
        self.n_ad = len(self.pos_ads)
        self.Z_ads = 0

        alpha_scale = 3.2
        gcut_scale = 1.0
        self.alpha = alpha_scale / self.rcut
        self.gcut = gcut_scale * self.alpha
        self.step = 1.0 * angstrom

        #self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, self.charges, self.alpha, self.gcut)
        self.e_el_real = 0
        self.e_vdw = 0
