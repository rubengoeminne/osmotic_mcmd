
import numpy as np
cimport numpy as np


cdef extern from "forceparts.h":
    void test();
    double ss_ei_full(int, double*, int, int, double*, double*, double*, double, double, double);
    double ss_ei_realspace(int, double*, int, int, double*, double*, double*, double, double, double);
    double ss_ei_realspace_insert(int, double*, int, int, double*, double*, double*, double, double, double);
    double mm3(int, double*, int, int, double*, double*, double*, double);
    double mm3_insert(int, int, int, double*, double*, double*, double*, double);
    double lj(int, double*, int, int, double*, double*, double*, double);
    double lj_insert(int, int, int, double*, double*, double*, double*, double);



def electrostatics(np.ndarray[double, ndim=2] pos, int Nframe, int Z_ads, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, np.ndarray[double, ndim=1] radii, double rcut, double alpha, double gcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == N
    assert radii.flags['C_CONTIGUOUS']
    assert radii.shape[0] == N
    assert rvecs.flags['C_CONTIGUOUS']
    assert rcut > 0

    return ss_ei_full(N, <double*>pos.data, Nframe, Z_ads, <double*>rvecs.data, <double*>charges.data, <double*>radii.data, rcut, alpha, gcut)


def electrostatics_realspace(np.ndarray[double, ndim=2] pos, int Nframe, int Z_ads, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, np.ndarray[double, ndim=1] radii, double rcut, double alpha, double gcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == N
    assert rcut > 0

    return ss_ei_realspace(N, <double*>pos.data, Nframe, Z_ads, <double*>rvecs.data, <double*>charges.data, <double*>radii.data, rcut, alpha, gcut)


def electrostatics_realspace_insert(int Nframe, int Nold, np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, np.ndarray[double, ndim=1] radii, double rcut, double alpha, double gcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == N
    assert radii.flags['C_CONTIGUOUS']
    assert radii.shape[0] == N
    assert rcut > 0

    return ss_ei_realspace_insert(N, <double*>pos.data, Nframe, Nold, <double*>rvecs.data, <double*>charges.data, <double*>radii.data, rcut, alpha, gcut)


def MM3(np.ndarray[double, ndim=2] pos, int Nframe, int Z_ads, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] sigmas, np.ndarray[double, ndim=1] epsilons, double rcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert sigmas.flags['C_CONTIGUOUS']
    assert sigmas.shape[0] == N
    assert epsilons.flags['C_CONTIGUOUS']
    assert epsilons.shape[0] == N
    assert rcut > 0

    return mm3(N, <double*>pos.data, Nframe, Z_ads, <double*>rvecs.data, <double*>sigmas.data, <double*>epsilons.data, rcut)


def MM3_insert(np.ndarray[double, ndim=2] pos, int Nold, int Nframe, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] sigmas, np.ndarray[double, ndim=1] epsilons, double rcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert sigmas.flags['C_CONTIGUOUS']
    assert sigmas.shape[0] == N
    assert epsilons.flags['C_CONTIGUOUS']
    assert epsilons.shape[0] == N
    assert rcut > 0

    return mm3_insert(N, Nold, Nframe, <double*>pos.data, <double*>rvecs.data, <double*>sigmas.data, <double*>epsilons.data, rcut)


def LJ(np.ndarray[double, ndim=2] pos, int Nframe, int Z_ads, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] sigmas, np.ndarray[double, ndim=1] epsilons, double rcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert sigmas.flags['C_CONTIGUOUS']
    assert sigmas.shape[0] == N
    assert epsilons.flags['C_CONTIGUOUS']
    assert epsilons.shape[0] == N
    assert rcut > 0

    return lj(N, <double*>pos.data, Nframe, Z_ads, <double*>rvecs.data, <double*>sigmas.data, <double*>epsilons.data, rcut)


def LJ_insert(np.ndarray[double, ndim=2] pos, int Nold, int Nframe, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] sigmas, np.ndarray[double, ndim=1] epsilons, double rcut):
    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert sigmas.flags['C_CONTIGUOUS']
    assert sigmas.shape[0] == N
    assert epsilons.flags['C_CONTIGUOUS']
    assert epsilons.shape[0] == N
    assert rcut > 0

    return lj_insert(N, Nold, Nframe, <double*>pos.data, <double*>rvecs.data, <double*>sigmas.data, <double*>epsilons.data, rcut)






