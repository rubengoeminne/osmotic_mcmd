"""
mcmd.py
A package to perform simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine

Handles the primary functions
"""

import os, sys
import numpy as np
from copy import deepcopy
from molmod.constants import boltzmann
from molmod.constants import planck as h
from molmod.units import angstrom, kjmol, kelvin, femtosecond, bar, kcalmol
from time import time
from scipy.spatial import cKDTree
import h5py as h5

from osmotic_mcmd.utilities import Acceptance, Parse_data, random_ads, random_rot
from wrapper_ewald import Sfac, ewald_insertion, ewald_deletion, ewald_displace, ewald_from_sfac
from wrapper_forceparts import electrostatics, electrostatics_realspace, electrostatics_realspace_insert, MM3, MM3_insert, LJ, LJ_insert



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

        assert data.ei == True
        assert data.mm3 == True

        self.pos = data.pos_MOF
        self.rvecs = data.rvecs
        self.V = np.linalg.det(self.rvecs)
        self.rvecs_flat = self.rvecs.reshape(9)
        self.N_frame = len(self.pos)
        self.nads = len(data.pos_ads)
        self.charges = data.charges_MOF
        self.pos_ads = data.pos_ads
        self.n_ad = len(self.pos_ads)
        self.Z_ads = 0
        self.write_trajectory = write_trajectory

        alpha_scale = 3.2
        gcut_scale = 1.0
        self.alpha = alpha_scale / self.rcut
        self.gcut = gcut_scale * self.alpha
        self.step = 1.0 * angstrom

        self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, self.charges, self.alpha, self.gcut)
        self.e_el_real = 0
        self.e_vdw = 0


    def overlap(self, pos):
        tree = cKDTree(pos, compact_nodes=False, copy_data=False, balanced_tree=False)
        pairs = tree.query_pairs(0.75*angstrom)
        return len(list(pairs)) > 0


    def compute_insertion(self, new_pos):
        n = self.Z_ads
        plen = len(self.pos)

        self.sfac = np.array(ewald_insertion(self.sfac, new_pos, self.rvecs_flat, self.data.charges_ads, self.alpha, self.gcut))
        e_ewald = ewald_from_sfac(self.sfac, self.rvecs_flat, self.alpha, self.gcut)
        self.e_el_real += electrostatics_realspace_insert(self.N_frame, len(self.pos)-self.nads, self.pos, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)

        if(self.data.mm3):
            self.e_vdw += MM3_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            self.e_vdw += LJ_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_ewald + self.e_el_real + self.e_vdw


    def compute_deletion(self, deleted_coord):
        n = self.Z_ads
        new_pos = np.append(self.pos, deleted_coord, axis=0)
        plen = len(new_pos)

        self.sfac = np.array(ewald_deletion(self.sfac, deleted_coord, self.rvecs_flat, self.data.charges_ads, self.alpha, self.gcut))
        e_ewald = ewald_from_sfac(self.sfac, self.rvecs_flat, self.alpha, self.gcut);
        self.e_el_real -= electrostatics_realspace_insert(self.N_frame, len(self.pos), new_pos, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)

        if(self.data.mm3):
            self.e_vdw -= MM3_insert(new_pos, len(self.pos), self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            self.e_vdw -= LJ_insert(new_pos, len(self.pos), self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_ewald + self.e_el_real + self.e_vdw


    def compute(self):
        n = self.Z_ads
        plen = len(self.pos)

        t = time()
        if(n == 0):
            return 0

        e_el = electrostatics(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)

        if(self.data.mm3):
            e_vdw = MM3(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            e_vdw = LJ(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_el + e_vdw


    def insertion(self, new_pos):
        self.Z_ads += 1

        self.pos = np.append(self.pos, new_pos, axis=0)
        e_new = self.compute_insertion(new_pos)

        return e_new


    def deletion(self):
        iatom = np.random.randint(self.Z_ads)
        index = self.N_frame + self.nads*iatom
        self.Z_ads -= 1

        deleted_coord = deepcopy(self.pos[index:index+self.nads])
        self.pos = np.delete(deepcopy(self.pos), np.s_[index:index+self.nads], axis=0)
        e_new = self.compute_deletion(deleted_coord)

        return deleted_coord, e_new


    def run_GCMC(self, N_iterations, N_sample):

        A = Acceptance()

        if not (os.path.isdir('results')):
            os.mkdir('results')

        e = 0
        t_it = time()

        N_samples = []
        E_samples = []
        pressures = []

        print('\n Iteration  inst. N    inst. E    time [s]')
        print('--------------------------------------------')

        for iteration in range(N_iterations+1):

            sfac_init = deepcopy(self.sfac)
            pos_init = deepcopy(self.pos)
            e_el_real_init = self.e_el_real
            e_vdw_init = self.e_vdw
            switch = np.random.rand()
            acc = 0

            # Insertion / deletion
            if(switch < self.prob[0]):

                if(switch < self.prob[0]/2):

                    new_pos = random_ads(self.pos_ads, self.rvecs)
                    if not self.overlap(np.append(self.pos, new_pos, axis=0)):
                        e_new = self.insertion(new_pos)
                    else:
                        self.Z_ads += 1
                        e_new = 10e5

                    exp_value = self.beta * (-e_new + e)
                    if(exp_value > 100):
                        acc = 1
                    elif(exp_value < -100):
                        acc = 0
                    else:
                        acc = min(1, self.V*self.beta*self.fugacity/self.Z_ads * np.exp(exp_value))

                    # Reject monte carlo move
                    if np.random.rand() > acc:
                        self.pos = pos_init
                        self.sfac = sfac_init
                        self.e_el_real = e_el_real_init
                        self.e_vdw = e_vdw_init
                        self.Z_ads -= 1
                    else:
                        e = e_new
#                        print('Acc: ', e_new/kjmol)

                elif(self.Z_ads > 0):

                    deleted_coord, e_new = self.deletion()

                    exp_value = -self.beta * (e_new - e)
                    if(exp_value > 100):
                        acc = 1
                    else:
                        acc = min(1, (self.Z_ads+1)/self.V/self.beta/self.fugacity * np.exp(exp_value))

                    # Reject monte carlo move
                    if np.random.rand() > acc:
                        #print('reject delet')
                        self.pos = pos_init
                        self.sfac = sfac_init
                        self.e_el_real = e_el_real_init
                        self.e_vdw = e_vdw_init
                        self.Z_ads += 1
                    else:
                        e = e_new
#                        print('Del: ', e_new/kjmol)

            elif(switch < self.prob[1]):

                if self.Z_ads == 0: continue

                trial = np.random.randint(self.Z_ads)

                if(switch < self.prob[0] + (self.prob[1]-self.prob[0])/2):

                    # Calculate translation energy as deletion + insertion of molecule
                    deleted_coord, e_new = self.deletion()
                    deleted_coord += self.step * (np.random.rand(3) - 0.5)
                    e_new = self.insertion(deleted_coord)

                elif self.nads > 1:

                    # Calculate rotation energy as deletion + insertion of molecule
                    deleted_coord, e_new = self.deletion()
                    deleted_coord = random_rot(deleted_coord, circlefrac=0.05)
                    e_new = self.insertion(deleted_coord)

                exp_value = -self.beta * (e_new - e)
                if(exp_value > 0):
                    exp_value = 0
                acc = min(1, np.exp(exp_value))

                # Reject monte carlo move
                if np.random.rand() > acc:
                    self.pos = pos_init
                    self.sfac = sfac_init
                    self.e_el_real = e_el_real_init
                    self.e_vdw = e_vdw_init
                else:
                    e = e_new
#                    print('Trans: ', e_new/kjmol)

            else:

                # MD run
                print('e before MD: ', e/kjmol)
                from yaff import System, ForceField, XYZWriter, VerletScreenLog, MTKBarostat, \
                                 NHCThermostat, TBCombination, VerletIntegrator, HDF5Writer, bar

                n = np.append(self.data.numbers_MOF, np.tile(self.data.numbers_ads, self.Z_ads))

                ffa_MOF = self.data.system.ffatypes[self.data.system.ffatype_ids]
                ffa_ads = self.data.system_ads.ffatypes[self.data.system_ads.ffatype_ids]
                ffa = np.append(ffa_MOF, np.tile(ffa_ads, self.Z_ads))
                assert len(self.pos) == len(ffa)

                s = System(n, self.pos, ffatypes = ffa, rvecs=self.rvecs)
                s.detect_bonds()
                ff = ForceField.generate(s, self.ff_file, rcut=self.rcut, tailcorrections=True)

                vsl = VerletScreenLog(step=100)
                mtk = MTKBarostat(ff, temp=self.T, press=self.P, \
                        timecon=1000*femtosecond, vol_constraint = True, anisotropic = True)
                nhc = NHCThermostat(temp=self.T, timecon=100*femtosecond, chainlength=3)
                tbc = TBCombination(nhc, mtk)

                t = time()
                verlet = VerletIntegrator(ff, 0.5*femtosecond, hooks=[tbc, vsl])
                verlet.run(500)
                print('MD time: ', time()-t)

                # Rebuild data for MC
                pos_total = ff.system.pos
                self.pos = pos_total[:self.N_frame]
                pos_molecules = pos_total[self.N_frame:]

                self.rvecs = ff.system.cell.rvecs
                self.rvecs_flat = self.rvecs.reshape(9)
                self.V = np.linalg.det(self.rvecs)

                self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, \
                                    self.charges, self.alpha, self.gcut)
                self.e_el_real = 0
                self.e_vdw = 0

                if self.Z_ads > 0:
                    for p in np.split(pos_molecules, self.Z_ads):
                        e_new = self.insertion(p)
                        self.Z_ads -= 1
                    e = e_new
                else:
                    e = 0

                print('e after MD: ', e/kjmol)

            if(iteration % N_sample == 0 and iteration > 0):
                eprint = e
                if np.abs(eprint) < 1e-10:
                    eprint = 0
                print(' {:7.7}       {:7.7} {:7.7}    {:7.4}'.format(
                      str(iteration),str(self.Z_ads),str(eprint/kjmol), time()-t_it)
                      )
                t_it = time()
                N_samples.append(self.Z_ads)
                E_samples.append(e)

        print('Average N: %.3f'%np.average(N_samples))
        np.save('results/N_%.8f.npy'%(self.P/(3.3989315828e-09)), np.array(N_samples))
        np.save('results/E_%.8f.npy'%(self.P/(3.3989315828e-09)), np.array(E_samples))

        from yaff import System
        n = np.append(self.data.numbers_MOF, np.tile(self.data.numbers_ads, self.Z_ads))
        s = System(n, self.pos, rvecs=self.rvecs)
        s.to_file('results/end_%.8f.xyz'%(self.P/(3.3989315828e-09)))
