

#include "cell.h"

double ewald(int N, double* pos, int Nframe, cell_type* cell, double* charges, double alpha, double gcut);
int Sfactor_len(double* rvecs, double gcut);
double* Sfactors(int Sfac_len, int N, double* pos, int Nframe, double* rvecs, double* charges, double alpha, double gcut);
double ewald_from_Sfac(double* Sfac, double* rvecs, double alpha, double gcut);

void ewald_displ(double* Sfac, int n, double* pos_old, double* pos_new, double* rvecs, double* charges, double alpha, double gcut);
void ewald_insert(double* Sfac, int n, double* pos, double* rvecs, double* charges, double alpha, double gcut);
void ewald_delete(double* Sfac, int n, double* pos, double* rvecs, double* charges, double alpha, double gcut);













