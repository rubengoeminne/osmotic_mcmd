
#include "cell.h"


double ss_ei_real(double* delta, double q1, double q2, double radius1, double radius2, cell_type* cell, double rcut, double alpha);
double ss_ei_full(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut);

double ss_ei_realspace(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut);
double ss_ei_realspace_insert(int N, double* pos, int Nframe, int Nold, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut);

double mm3(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* sigmas, double* epsilons, double rcut);
double mm3_insert(int N, int Nold, int Nframe, double* pos, double* rvecs, double* sigmas, double* epsilons, double rcut);

double lj(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* sigmas, double* epsilons, double rcut);
double lj_insert(int N, int Nold, int Nframe, double* pos, double* rvecs, double* sigmas, double* epsilons, double rcut);


