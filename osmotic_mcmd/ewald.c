

#include <stdlib.h>
#include <math.h>
#include "cell.h"
#include "ewald.h"
#include <stdio.h>



double ewald(int N, double* pos, int Nframe, cell_type* cell, double* charges, double alpha, double gcut){
    double result = 0;

    double cosfac, sinfac, cosfacframe, sinfacframe, x, c, s;
    double j[3];
    // Precomput some factors
    double fac1 = 4.0*M_PI/cell->volume;
    double fac2 = 0.25/alpha/alpha;

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    // Reciprocal-space terms
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                cosfac = 0.0; sinfac = 0.0;
                cosfacframe = 0.0; sinfacframe = 0.0;
                for (int iatom=0; iatom<N; iatom++) {
                    x = k[0]*pos[3*iatom] + k[1]*pos[3*iatom+1] + k[2]*pos[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac += c;
                    sinfac += s;
                    if(iatom < Nframe){
                        cosfacframe += c;
                        sinfacframe += s;
                    }
                }
                c = fac1*exp(-ksq*fac2)/ksq;
                s = cosfac*cosfac + sinfac*sinfac - cosfacframe*cosfacframe - sinfacframe*sinfacframe;
                result += c*s;
            }
        }
    }
    //std::cout<<"reci: "<<result/(0.00038087991760)<<std::endl;
    return result;
}



int Sfactor_len(double* rvecs, double gcut){

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    // Determine the ranges of the reciprocal sum
    double kmax[3], j[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    // Determine the length of the Sfac array
    int Sfac_len = 0;
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;
                Sfac_len += 4;                
            }
        }
    }

    cell_free(cell);

    return Sfac_len;

}



double* Sfactors(int Sfac_len, int N, double* pos, int Nframe, double* rvecs, double* charges, double alpha, double gcut){

    double* Sfac = (double*) malloc(Sfac_len * sizeof(double));

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    double cosfac, sinfac, cosfacframe, sinfacframe, x, c, s;
    double j[3];

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    // Reciprocal-space terms
    int Sfac_counter = 0;
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                cosfac = 0.0; sinfac = 0.0;
                cosfacframe = 0.0; sinfacframe = 0.0;
                for (int iatom=0; iatom<N; iatom++) {
                    x = k[0]*pos[3*iatom] + k[1]*pos[3*iatom+1] + k[2]*pos[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac += c;
                    sinfac += s;
                    if(iatom < Nframe){
                        cosfacframe += c;
                        sinfacframe += s;
                    }
                }

                Sfac[Sfac_counter] = cosfac;
                Sfac[Sfac_counter+1] = sinfac;
                Sfac[Sfac_counter+2] = cosfacframe;
                Sfac[Sfac_counter+3] = sinfacframe;
                Sfac_counter += 4;
            }
        }
    }

    cell_free(cell);

    return Sfac;

}



double ewald_from_Sfac(double* Sfac, double* rvecs, double alpha, double gcut){
    double result = 0;

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    double c, s;
    double j[3];
    // Precomput some factors
    double fac1 = 4.0*M_PI/cell->volume;
    double fac2 = 0.25/alpha/alpha;

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    int Sfac_counter = 0;

    // Reciprocal-space terms
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                //std::cout<<Sfac_counter<<" "<<Sfac_counter*4<<std::endl;

                c = fac1*exp(-ksq*fac2)/ksq;
                s = Sfac[4*Sfac_counter]*Sfac[4*Sfac_counter] + Sfac[4*Sfac_counter+1]*Sfac[4*Sfac_counter+1];
                s -= Sfac[4*Sfac_counter+2]*Sfac[4*Sfac_counter+2] + Sfac[4*Sfac_counter+3]*Sfac[4*Sfac_counter+3];
                result += c*s;

                Sfac_counter += 1;
            }
        }
    }

    cell_free(cell);

    return result;
}



void ewald_displ(double* Sfac, int n, double* pos_old, double* pos_new, double* rvecs, double* charges, double alpha, double gcut){

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    double cosfac, sinfac, x, c, s;
    double j[3];

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    int Sfac_counter = 0;

    // Reciprocal-space terms
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                cosfac = 0.0; sinfac = 0.0;
                for (int iatom=0; iatom<n; iatom++) {
                    x = k[0]*pos_new[3*iatom] + k[1]*pos_new[3*iatom+1] + k[2]*pos_new[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac += c;
                    sinfac += s;

                    x = k[0]*pos_old[3*iatom] + k[1]*pos_old[3*iatom+1] + k[2]*pos_old[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac -= c;
                    sinfac -= s;
                }

                Sfac[4*Sfac_counter] += cosfac;
                Sfac[4*Sfac_counter+1] += sinfac;

                Sfac_counter += 1;

            }
        }
    }

    cell_free(cell);

    //return ewald_from_Sfac(Sfac, rvecs, alpha, gcut_orig);
}



void ewald_insert(double* Sfac, int n, double* pos, double* rvecs, double* charges, double alpha, double gcut){

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    double cosfac, sinfac, x, c, s;
    double j[3];

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    int Sfac_counter = 0;

    // Reciprocal-space terms
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                cosfac = 0.0; sinfac = 0.0;
                for (int iatom=0; iatom<n; iatom++) {
                    x = k[0]*pos[3*iatom] + k[1]*pos[3*iatom+1] + k[2]*pos[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac += c;
                    sinfac += s;
                }
                Sfac[4*Sfac_counter] += cosfac;
                Sfac[4*Sfac_counter+1] += sinfac;

                Sfac_counter += 1;

            }
        }
    }

    cell_free(cell);

    //return ewald_from_Sfac(Sfac, rvecs, alpha, gcut_orig);
}




void ewald_delete(double* Sfac, int n, double* pos, double* rvecs, double* charges, double alpha, double gcut){

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    double cosfac, sinfac, x, c, s;
    double j[3];

    // Determine the ranges of the reciprocal sum
    double kmax[3];
    kmax[0] = ceil(gcut/cell->gspacings[0]);
    kmax[1] = ceil(gcut/cell->gspacings[1]);
    kmax[2] = ceil(gcut/cell->gspacings[2]);
    gcut *= 2*M_PI;
    gcut *= gcut;

    int Sfac_counter = 0;

    // Reciprocal-space terms
    for (int j0=-kmax[0]; j0 <= kmax[0]; j0++) {
        j[0] = 2*M_PI*j0;
        for (int j1=-kmax[1]; j1 <= kmax[1]; j1++) {
            j[1] = 2*M_PI*j1;
            for (int j2=0; j2 <= kmax[2]; j2++) {
                if (j2==0) {
                    if (j1<0) continue;
                    if ((j1==0)&&(j0<=0)) continue;
                }
                j[2] = 2*M_PI*j2;
                double k[3];
                g_lincomb(cell, j, k);
                double ksq = k[0]*k[0] + k[1]*k[1] + k[2]*k[2];
                if (ksq > gcut) continue;

                cosfac = 0.0; sinfac = 0.0;
                for (int iatom=0; iatom<n; iatom++) {
                    x = k[0]*pos[3*iatom] + k[1]*pos[3*iatom+1] + k[2]*pos[3*iatom+2];
                    c = charges[iatom]*cos(x);
                    s = charges[iatom]*sin(x);
                    cosfac -= c;
                    sinfac -= s;
                }

                Sfac[4*Sfac_counter] += cosfac;
                Sfac[4*Sfac_counter+1] += sinfac;

                Sfac_counter += 1;
            }
        }
    }

    cell_free(cell);

    //return ewald_from_Sfac(Sfac, rvecs, alpha, gcut_orig);
}








