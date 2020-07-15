
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include "ewald.h"
#include "forceparts.h"




double get_radius(double radius1, double radius2){

    if(abs(radius1) < 1e-8 && abs(radius2) < 1e-8){
        return 0.0;
    }
    else if(abs(radius1) < 1e-8){
        return radius2;
    }
    else if(abs(radius2) < 1e-8){
        return radius1;
    }
    else
        return sqrt(radius1*radius1+radius2*radius2);
}



// Real space electrostatic ss interaction minus gaussian screening charge
double ss_ei_real(double* delta, double q1, double q2, double radius1, double radius2, cell_type* cell, double rcut, double alpha){
    double result = 0;

    // Determine the ranges of the real sum
    long rbegin[3], rend[3];

    cell_set_ranges_rcut(cell, delta, rcut, rbegin, rend);

    double d, d2, radius;
    double rcut2 = rcut*rcut;

    radius = get_radius(radius1, radius2);

    // Loop over all appropriate images
    double j[3];
    for (int j0=rbegin[0]; j0 < rend[0]; j0++) {
        j[0] = j0;
        for (int j1=rbegin[1]; j1 < rend[1]; j1++) {
            j[1] = j1;
            for (int j2=rbegin[2]; j2 < rend[2]; j2++) {
                j[2] = j2;
                double tmp[3];
                cell_to_cart(cell, j, tmp);
                tmp[0] += delta[0];
                tmp[1] += delta[1];
                tmp[2] += delta[2];
                d2 = tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2];
                if(d2<rcut2 && d2!=0.0){
                    d = sqrt(d2);
                    if(radius > 1e-8)
                        result += q1*q2*(erf(d/radius)-erf(alpha*d))/d;
                    else
                        result += q1*q2*(1-erf(alpha*d))/d;
                }
            }
        }
    }
    return result;

}


// Total electrostatic ss interaction energy per unit cell
double ss_ei_full(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut){
    double result = 0;

    double qtot = 0;
    for (int ch_it=0;ch_it<N;ch_it++){
        qtot += charges[ch_it];
    }

    double delta[3];
    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    //
    // REAL SPACE TERMS
    //

    for (int i=0;i<N;i++){
        for (int j=Nframe;j<N;j++){
            delta[0] = pos[3*j]   - pos[3*i];
            delta[1] = pos[3*j+1] - pos[3*i+1];
            delta[2] = pos[3*j+2] - pos[3*i+2];
            if(i>=Nframe)
                result += 0.5*ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
            else
                result += ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
        }
    }

    // Subtract intermolecular contributions
    double Nmol;
    if(Z_ads > 0)
        Nmol = (int)((N-Nframe)/Z_ads);
    else
        Nmol = 0;

    double mol_iter, d, radius;
    for (int i=Nframe;i<N;i++){
        mol_iter = floor((i-Nframe)/Nmol);
        for(int j=Nframe + mol_iter*Nmol;j<Nframe + (mol_iter+1)*Nmol;j++){
            if(i!=j){
                radius = get_radius(radii[i], radii[j]);
                delta[0] = pos[3*j] - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];
                d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
                if(radius > 1e-8)
                    result -= 0.5*charges[i]*charges[j]*erf(d/radius)/d;
                else
                    result -= 0.5*charges[i]*charges[j]/d;
            }
        }
    }

    //
    // SELFINTERACTION TERMS
    //

    double self = 0;
    for (int i=Nframe;i<N;i++){
        self -= alpha/sqrt(M_PI)*charges[i]*charges[i];
    }
    result += self;

    //
    // RECIPROCAL SPACE TERMS
    //

    result += ewald(N, pos, Nframe, cell, charges, alpha, gcut);

    //
    // NEUTRALIZING BACKGROUND
    //

    cell_free(cell);

    result += M_PI*qtot*qtot/2/cell->volume/alpha/alpha;
    return result;

}



// Total electrostatic ss interaction energy per unit cell
double ss_ei_realspace(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut){
    double result = 0;

    double qads = 0;
    for (int ch_it=Nframe;ch_it<N;ch_it++){
        qads += charges[ch_it];
    }

    double delta[3];
    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    //
    // REAL SPACE TERMS
    //

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta)
    for (int i=0;i<N;i++){
        for (int j=Nframe;j<N;j++){
            delta[0] = pos[3*j]   - pos[3*i];
            delta[1] = pos[3*j+1] - pos[3*i+1];
            delta[2] = pos[3*j+2] - pos[3*i+2];
            if(i>=Nframe)
                result += 0.5*ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
            else{
                result += ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
            }
        }
    }

    // Subtract intermolecular contributions
    double Nmol;
    if(Z_ads > 0)
        Nmol = (int)((N-Nframe)/Z_ads);
    else{
        Nmol = 0;
        cell_free(cell);
        return result;
    }

    double mol_iter, d, radius;
    for (int i=Nframe;i<N;i++){
        mol_iter = floor((i-Nframe)/Nmol);
        for(int j=Nframe + mol_iter*Nmol;j<Nframe + (mol_iter+1)*Nmol;j++){
            if(i!=j){
                radius = get_radius(radii[i], radii[j]);
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];
                d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
                if(radius > 1e-8){
                    result -= 0.5*charges[i]*charges[j]*erf(d/radius)/d;
                }
                else
                    result -= 0.5*charges[i]*charges[j]/d;
            }
        }
    }


    //
    // SELFINTERACTION TERMS
    //

    /*
    for (int i=Nframe;i<N;i++){
        result -= alpha/sqrt(M_PI)*charges[i]*charges[i];
    }
    */

    //
    // NEUTRALIZING BACKGROUND
    //

    /*
    result += M_PI*qads*qads/2/cell->volume/alpha/alpha;
    */

    cell_free(cell);

    return result;
}



// Total electrostatic ss interaction energy per unit cell
double ss_ei_realspace_insert(int N, double* pos, int Nframe, int Nold, double* rvecs, double* charges, double* radii, double rcut, double alpha, double gcut){
    double result = 0;

    double qtot, qprev;
    qtot = 0; qprev = 0;
    for (int ch_it=Nold;ch_it<N;ch_it++){
        qtot += charges[ch_it];
        if(ch_it < Nold)
            qprev += charges[ch_it];
    }

    double delta[3];
    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

    //
    // REAL SPACE TERMS
    //

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta)
    for (int i=0;i<N;i++){
        for (int j=Nold;j<N;j++){
            delta[0] = pos[3*j]   - pos[3*i];
            delta[1] = pos[3*j+1] - pos[3*i+1];
            delta[2] = pos[3*j+2] - pos[3*i+2];
            if(i>=Nold)
                result += 0.5*ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
            else
                result += ss_ei_real(delta, charges[i], charges[j], radii[i], radii[j], cell, rcut, alpha);
        }
    }

    // Subtract intermolecular contributions
    double d, radius;
    for (int i=Nold;i<N;i++){
        for(int j=Nold;j<N;j++){
            if(i!=j){
                radius = get_radius(radii[i], radii[j]);
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];
                d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
                if(radius > 1e-8)
                    result -= 0.5*charges[i]*charges[j]*erf(d/radius)/d;
                else
                    result -= 0.5*charges[i]*charges[j]/d;
            }
        }
    }

    //
    // SELFINTERACTION TERMS
    //

    /*
    double self = 0.0;
    for (int i=Nold;i<N;i++){
        self -= alpha/sqrt(M_PI)*charges[i]*charges[i];
    }
    result += self;
    */

    //
    // NEUTRALIZING BACKGROUND
    //

    /*
    double neut = 0.0;
    if(alpha > 0.0){
        neut += M_PI*(qtot*qtot-qprev*qprev)/2/cell->volume/alpha/alpha;
    }
    result += neut;
    */

    cell_free(cell);

    return result;
}




// Total dispersion ss interaction energy per unit cell
double mm3(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* sigmas, double* epsilons, double rcut){
    double result = 0;
    if(Z_ads == 0)
        return result;

    double delta[3]; long rbegin[3], rend[3]; double j[3]; double sigma, epsilon;

    double d, d2, s2_d2;
    double rcut2 = rcut*rcut;

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta, rbegin, rend, j, c6, c8, xAB, xAB_d, expx, d, d2) shared(rcut2)
    for (int iatom0=0;iatom0<N;iatom0++){
        for (int iatom1=Nframe;iatom1<N;iatom1++){
            delta[0] = pos[3*iatom1]   - pos[3*iatom0];
            delta[1] = pos[3*iatom1+1] - pos[3*iatom0+1];
            delta[2] = pos[3*iatom1+2] - pos[3*iatom0+2];

            sigma = sigmas[iatom0]+sigmas[iatom1];
            epsilon = sqrt(epsilons[iatom0]*epsilons[iatom1]);

//            if(iatom0>=Nframe)
//                result += -2*M_PI/cell->volume*(-2.25*epsilon/3*pow(sigma/rcut,3));
//            else
//                result += -4*M_PI/cell->volume*(-2.25*epsilon/3*pow(sigma/rcut,3));

            // Determine the ranges of the real sum
            cell_set_ranges_rcut(cell, delta, rcut, rbegin, rend);
            // Loop over all appropriate images
            for (int j0=rbegin[0]; j0 < rend[0]; j0++) {
                j[0] = j0;
                for (int j1=rbegin[1]; j1 < rend[1]; j1++) {
                    j[1] = j1;
                    for (int j2=rbegin[2]; j2 < rend[2]; j2++) {
                        j[2] = j2;
                        double tmp[3];
                        cell_to_cart(cell, j, tmp);
                        tmp[0] += delta[0];
                        tmp[1] += delta[1];
                        tmp[2] += delta[2];
                        d2 = tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2];
                        if(d2<rcut2 && d2!=0.0){
                            d = sqrt(d2);
                            if(d < 0.355114 * sigma){
                                d = 0.355114 * sigma;
                                d2 = d*d;
                            }
                            s2_d2 = sigma*sigma/d2;
                            if(iatom0>=Nframe)
                                result += 0.5*epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                            else
                                result += epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                        }
                    }
                }
            }
        }
    }


    // Subtract intermolecular contributions
    double Nmol = (int)((N-Nframe)/Z_ads);
    double mol_iter;
    for (int i=Nframe;i<N;i++){
        mol_iter = floor((i-Nframe)/Nmol);
        for(int j=Nframe + mol_iter*Nmol;j<Nframe + (mol_iter+1)*Nmol;j++){
            if(i!=j){
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];

                sigma = sigmas[i]+sigmas[j];
                epsilon = sqrt(epsilons[i]*epsilons[j]);

                d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if(d2<rcut2 && d2!=0.0){
                    d = sqrt(d2);
                    if(d < 0.355114 * sigma){
                        d = 0.355114 * sigma;
                        d2 = d*d;
                    }
                    s2_d2 = sigma*sigma/d2;
                    result -= 0.5*epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                }

            }
        }
    }

    cell_free(cell);

    return result;
}



// Total dispersion ss interaction energy per unit cell
double mm3_insert(int N, int Nold, int Nframe, double* pos, double* rvecs, double* sigmas, double* epsilons, double rcut){
    double result = 0;
    double delta[3]; long rbegin[3], rend[3]; double j[3]; double sigma, epsilon;

    double d, d2, s2_d2;
    double rcut2 = rcut*rcut;

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta, rbegin, rend, j, c6, c8, xAB, xAB_d, expx, d, d2) shared(rcut2)
    for (int iatom0=0;iatom0<N;iatom0++){
        for (int iatom1=Nold;iatom1<N;iatom1++){
            delta[0] = pos[3*iatom1]   - pos[3*iatom0];
            delta[1] = pos[3*iatom1+1] - pos[3*iatom0+1];
            delta[2] = pos[3*iatom1+2] - pos[3*iatom0+2];

            sigma = sigmas[iatom0]+sigmas[iatom1];
            epsilon = sqrt(epsilons[iatom0]*epsilons[iatom1]);

//            if(iatom0>=Nold)
//                result += -2*M_PI/cell->volume*(-2.25*epsilon/3*pow(sigma/rcut,3));
//            else
//                result += -4*M_PI/cell->volume*(-2.25*epsilon/3*pow(sigma/rcut,3));

            // Determine the ranges of the real sum
            cell_set_ranges_rcut(cell, delta, rcut, rbegin, rend);
            // Loop over all appropriate images
            for (int j0=rbegin[0]; j0 < rend[0]; j0++) {
                j[0] = j0;
                for (int j1=rbegin[1]; j1 < rend[1]; j1++) {
                    j[1] = j1;
                    for (int j2=rbegin[2]; j2 < rend[2]; j2++) {
                        j[2] = j2;
                        double tmp[3];
                        cell_to_cart(cell, j, tmp);
                        tmp[0] += delta[0];
                        tmp[1] += delta[1];
                        tmp[2] += delta[2];
                        d2 = tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2];
                        if(d2<rcut2 && d2!=0.0){
                            d = sqrt(d2);
                            if(d < 0.355114 * sigma){
                                d = 0.355114 * sigma;
                                d2 = d*d;
                            }
                            s2_d2 = sigma*sigma/d2;
                            if(iatom0>=Nold)
                                result += 0.5*epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                            else
                                result += epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                        }
                    }
                }
            }
        }
    }

    // Subtract intermolecular contributions
    for (int i=Nold;i<N;i++){
        for(int j=Nold;j<N;j++){
            if(i!=j){
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];

                sigma = sigmas[i]+sigmas[j];
                epsilon = sqrt(epsilons[i]*epsilons[j]);

                d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if(d2<rcut2 && d2!=0.0){
                    d = sqrt(d2);
                    if(d < 0.355114 * sigma){
                        d = 0.355114 * sigma;
                        d2 = d*d;
                    }
                    s2_d2 = sigma*sigma/d2;
                    result -= 0.5*epsilon*(184000*exp(-12*d/sigma)-2.25*s2_d2*s2_d2*s2_d2);
                }

            }
        }
    }

    cell_free(cell);

    return result;
}





// Total dispersion ss interaction energy per unit cell
double lj(int N, double* pos, int Nframe, int Z_ads, double* rvecs, double* sigmas, double* epsilons, double rcut){
    double result = 0;
    if(Z_ads == 0)
        return result;

    double delta[3]; long rbegin[3], rend[3]; double j[3]; double sigma, epsilon;

    double d2, s2_d2, s6_d6;
    double rcut2 = rcut*rcut;

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta, rbegin, rend, j, c6, c8, xAB, xAB_d, expx, d, d2) shared(rcut2)
    for (int iatom0=0;iatom0<N;iatom0++){
        for (int iatom1=Nframe;iatom1<N;iatom1++){
            delta[0] = pos[3*iatom1]   - pos[3*iatom0];
            delta[1] = pos[3*iatom1+1] - pos[3*iatom0+1];
            delta[2] = pos[3*iatom1+2] - pos[3*iatom0+2];

            sigma = 0.5*(sigmas[iatom0]+sigmas[iatom1]);
            epsilon = sqrt(epsilons[iatom0]*epsilons[iatom1]);

            /*
            if(iatom0>=Nframe)
                result += -2*M_PI/cell->volume*(-4*epsilon/3*pow(sigma/rcut,3));
            else
                result += -4*M_PI/cell->volume*(-4*epsilon/3*pow(sigma/rcut,3));
            */

            // Determine the ranges of the real sum
            cell_set_ranges_rcut(cell, delta, rcut, rbegin, rend);
            // Loop over all appropriate images
            for (int j0=rbegin[0]; j0 < rend[0]; j0++) {
                j[0] = j0;
                for (int j1=rbegin[1]; j1 < rend[1]; j1++) {
                    j[1] = j1;
                    for (int j2=rbegin[2]; j2 < rend[2]; j2++) {
                        j[2] = j2;
                        double tmp[3];
                        cell_to_cart(cell, j, tmp);
                        tmp[0] += delta[0];
                        tmp[1] += delta[1];
                        tmp[2] += delta[2];
                        d2 = tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2];
                        if(d2<rcut2 && d2!=0.0){
                            s2_d2 = sigma*sigma/d2;
                            s6_d6 = s2_d2*s2_d2*s2_d2;
                            if(iatom0>=Nframe)
                                result += 0.5*4*epsilon*(s6_d6*(s6_d6-1));
                            else
                                result += 4*epsilon*(s6_d6*(s6_d6-1));
                        }
                    }
                }
            }
        }
    }


    // Subtract intermolecular contributions
    double Nmol = (int)((N-Nframe)/Z_ads);
    double mol_iter;
    for (int i=Nframe;i<N;i++){
        mol_iter = floor((i-Nframe)/Nmol);
        for(int j=Nframe + mol_iter*Nmol;j<Nframe + (mol_iter+1)*Nmol;j++){
            if(i!=j){
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];

                sigma = 0.5*(sigmas[i]+sigmas[j]);
                epsilon = sqrt(epsilons[i]*epsilons[j]);

                d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if(d2<rcut2 && d2!=0.0){
                    s2_d2 = sigma*sigma/d2;
                    s6_d6 = s2_d2*s2_d2*s2_d2;
                    result -= 0.5*4*epsilon*(s6_d6*(s6_d6-1));
                }
            }
        }
    }

    cell_free(cell);

    return result;
}




// Total dispersion ss interaction energy per unit cell
double lj_insert(int N, int Nold, int Nframe, double* pos, double* rvecs, double* sigmas, double* epsilons, double rcut){
    double result = 0;
    double delta[3]; long rbegin[3], rend[3]; double j[3]; double sigma, epsilon;

    double d2, s2_d2, s6_d6;
    double rcut2 = rcut*rcut;

    cell_type* cell = cell_new();
    cell_init(cell, rvecs, 3);

//#pragma omp parallel for schedule(dynamic, 1) reduction(+ : result) private(delta, rbegin, rend, j, c6, c8, xAB, xAB_d, expx, d, d2) shared(rcut2)
    for (int iatom0=0;iatom0<N;iatom0++){
        for (int iatom1=Nold;iatom1<N;iatom1++){
            delta[0] = pos[3*iatom1]   - pos[3*iatom0];
            delta[1] = pos[3*iatom1+1] - pos[3*iatom0+1];
            delta[2] = pos[3*iatom1+2] - pos[3*iatom0+2];

            sigma = 0.5*(sigmas[iatom0]+sigmas[iatom1]);
            epsilon = sqrt(epsilons[iatom0]*epsilons[iatom1]);

            if(iatom0>=Nold)
                result += -2*M_PI/cell->volume*(-4*epsilon/3*pow(sigma/rcut,3));
            else
                result += -4*M_PI/cell->volume*(-4*epsilon/3*pow(sigma/rcut,3));

            // Determine the ranges of the real sum
            cell_set_ranges_rcut(cell, delta, rcut, rbegin, rend);
            // Loop over all appropriate images
            for (int j0=rbegin[0]; j0 < rend[0]; j0++) {
                j[0] = j0;
                for (int j1=rbegin[1]; j1 < rend[1]; j1++) {
                    j[1] = j1;
                    for (int j2=rbegin[2]; j2 < rend[2]; j2++) {
                        j[2] = j2;
                        double tmp[3];
                        cell_to_cart(cell, j, tmp);
                        tmp[0] += delta[0];
                        tmp[1] += delta[1];
                        tmp[2] += delta[2];
                        d2 = tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2];
                        if(d2<rcut2 && d2!=0.0){
                            s2_d2 = sigma*sigma/d2;
                            s6_d6 = s2_d2*s2_d2*s2_d2;
                            if(iatom0>=Nold)
                                result += 0.5*4*epsilon*(s6_d6*(s6_d6-1));
                            else
                                result += 4*epsilon*(s6_d6*(s6_d6-1));
                        }
                    }
                }
            }
        }
    }

    // Subtract intermolecular contributions
    for (int i=Nold;i<N;i++){
        for(int j=Nold;j<N;j++){
            if(i!=j){
                delta[0] = pos[3*j]   - pos[3*i];
                delta[1] = pos[3*j+1] - pos[3*i+1];
                delta[2] = pos[3*j+2] - pos[3*i+2];

                sigma = 0.5*(sigmas[i]+sigmas[j]);
                epsilon = sqrt(epsilons[i]*epsilons[j]);

                d2 = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2];
                if(d2<rcut2 && d2!=0.0){
                    s2_d2 = sigma*sigma/d2;
                    s6_d6 = s2_d2*s2_d2*s2_d2;
                    result -= 0.5*4*epsilon*(s6_d6*(s6_d6-1));
                }

            }
        }
    }

    cell_free(cell);

    return result;
}














