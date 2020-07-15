

import numpy as np
cimport numpy as np


cdef extern from "ewald.h":
    int Sfactor_len(double*, double);
    double* Sfactors(int, int, double*, int, double*, double*, double, double);
    double ewald_from_Sfac(double*, double*, double, double);
    void ewald_displ(double*, int, double*, double*, double*, double*, double, double);
    void ewald_insert(double*, int, double*, double*, double*, double, double);
    void ewald_delete(double*, int, double*, double*, double*, double, double);


def Sfac(np.ndarray[double, ndim=2] pos, int Nframe, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, double alpha, double gcut):

    assert pos.flags['C_CONTIGUOUS']
    cdef int N = pos.shape[0]
    assert charges.flags['C_CONTIGUOUS']
    assert charges.shape[0] == N

    cdef int sfac_len = Sfactor_len(<double*>rvecs.data, gcut)
    cdef double* sfac = Sfactors(sfac_len, N, <double*>pos.data, Nframe, <double*>rvecs.data, <double*>charges.data, alpha, gcut)

    sfacarr = np.asarray(<double[:sfac_len]>sfac)
    return sfacarr


def ewald_from_sfac(np.ndarray[double, ndim=1] sfac, np.ndarray[double, ndim=1] rvecs, double alpha, double gcut):
    return ewald_from_Sfac(<double*>sfac.data, <double*>rvecs.data, alpha, gcut)


def ewald_displace(np.ndarray[double, ndim=1] sfac, np.ndarray[double, ndim=2] pos_old, np.ndarray[double, ndim=2] pos_new, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, double alpha, double gcut):
    cdef int n = pos_old.shape[0]
    assert n == pos_new.shape[0]

    cdef double* sfac_ = <double*>sfac.data

    ewald_displ(sfac_, n, <double*>pos_old.data, <double*>pos_new.data, <double*>rvecs.data, <double*>charges.data, alpha, gcut)

    return np.asarray(<double[:len(sfac)]>sfac_)


def ewald_insertion(np.ndarray[double, ndim=1] sfac, np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, double alpha, double gcut):
    cdef int n = pos.shape[0]

    cdef double* sfac_ = <double*>sfac.data

    ewald_insert(sfac_, n, <double*>pos.data, <double*>rvecs.data, <double*>charges.data, alpha, gcut)

    return np.asarray(<double[:len(sfac)]>sfac_)


def ewald_deletion(np.ndarray[double, ndim=1] sfac, np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] rvecs, np.ndarray[double, ndim=1] charges, double alpha, double gcut):
    cdef int n = pos.shape[0]
    assert n == pos.shape[0]

    cdef double* sfac_ = <double*>sfac.data

    ewald_delete(sfac_, n, <double*>pos.data, <double*>rvecs.data, <double*>charges.data, alpha, gcut)

    return np.asarray(<double[:len(sfac)]>sfac_)



