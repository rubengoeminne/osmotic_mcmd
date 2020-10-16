"""
utilities.py
Some utilities used by the main mcmd class related to the parsing of input data
and the randomization of adsorbant positions
"""

import numpy as np
from yaff import System, ForceField, log
from molmod.units import angstrom, kjmol, kcalmol, amu



class Acceptance():
    """Class to keep track of the acceptance ratio of different GCMC moves
    """
    def __init__(self):
        self.trans_steps = 0
        self.trans_acc = 0
        self.rot_steps = 0
        self.rot_acc = 0
        self.insertion_steps = 0
        self.insertion_acc = 0
        self.deletion_steps = 0
        self.deletion_acc = 0

    def get_acceptance_prob(self):
        acc = []
        if(self.trans_steps > 0):
            acc.append(float(self.trans_acc)/self.trans_steps)
        else:
            acc.append(0)

        if(self.rot_steps > 0):
            acc.append(float(self.rot_acc)/self.rot_steps)
        else:
            acc.append(0)

        if(self.insertion_steps > 0):
            acc.append(float(self.insertion_acc)/self.insertion_steps)
        else:
            acc.append(0)

        if(self.deletion_steps > 0):
            acc.append(float(self.deletion_acc)/self.deletion_steps)
        else:
            acc.append(0)

        return acc

#atomic_masses = np.array([0.0, 1.008, 4.003, 6.941, 9.012, 10.811, 12.011, 14.007, 15.999, 18.998, 20.18, 22.99, 24.305, 26.982, 28.086, 30.974, 32.065, 35.453, 39.948])*amu

class Parse_data():
    def __init__(self, system_file, adsorbate_file, ff_file, rcut):

        system = System.from_file(system_file)
        adsorbate = System.from_file(adsorbate_file)

        self.sigmas_MOF, self.epsilons_MOF, self.charges_MOF, self.radii_MOF = self.get_data(system, ff_file, rcut)
        self.sigmas_ads, self.epsilons_ads, self.charges_ads, self.radii_ads = self.get_data(adsorbate, ff_file, rcut)

        system.set_standard_masses()
        self.pos_MOF = system.pos
        self.rvecs = system.cell.rvecs
        self.numbers_MOF = system.numbers
        self.mass_MOF = sum(system.masses)
        self.system = system

        self.pos_ads = adsorbate.pos
        self.numbers_ads = adsorbate.numbers
#        self.mass_ads = sum(atomic_masses[self.numbers_ads])
        self.system_ads = adsorbate

        self.parameter_list()

    def get_data(self, system, ff_file, rcut):

        pos = system.pos
        rvecs = system.cell.rvecs
        ff = ForceField.generate(system, ff_file, rcut=rcut)

        self.lj = False
        self.mm3 = False
        self.ei = False
        for part in ff.parts:

            if('pair_mm3' in part.name or part.name == 'pair_lj'):
                sigmas = part.pair_pot.sigmas
                epsilons = part.pair_pot.epsilons
                if('pair_mm3' in part.name):
                    self.mm3 = True
                else:
                    self.lj = True

            if(part.name == 'pair_ei'):
                charges = part.pair_pot.charges
                radii = part.pair_pot.radii
                self.ei = True

        if not self.mm3 and not self.lj:
            print('No vdW loaded')
            sigmas = np.zeros(len(pos))
            epsilons = np.zeros(len(pos))

        if not self.ei:
            print('No EI loaded')
            charges = np.zeros(len(pos))
            radii = np.zeros(len(pos))

        return sigmas, epsilons, charges, radii

    # Keep lists of data for different numbers of adsorbates in memory for efficiency
    def parameter_list(self):

        self.sigmas = np.append(self.sigmas_MOF, np.tile(self.sigmas_ads, 2000))
        self.epsilons = np.append(self.epsilons_MOF, np.tile(self.epsilons_ads, 2000))
        self.charges = np.append(self.charges_MOF, np.tile(self.charges_ads, 2000))
        self.radii = np.append(self.radii_MOF, np.tile(self.radii_ads, 2000))



def random_rot(pos, circlefrac=1):

    # Translate to origin
    com = np.average(pos, axis=0)
    pos -= com

    # from yaff
    randnums = np.random.uniform(size=(3,))
    theta, phi, z = randnums

    theta = theta * 2.0*circlefrac*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*circlefrac  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)

    pos = np.einsum('ib,ab->ia', pos, M)

    return pos + com


def random_ads(pos, rvecs):

    if len(pos) != 0:
        pos = random_rot(pos)

    pos -= np.average(pos, axis=0)
    new_com = np.random.rand()*rvecs[0] + np.random.rand()*rvecs[1] + np.random.rand()*rvecs[2]

    return pos + new_com


