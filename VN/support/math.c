//
// Created by jonas on 13.05.22.
//

#include "math.h"
double x_to_the_power_of_n(double *pDouble, unsigned int n) {
    double result = 1;
    double p = *pDouble;
    for(int i = 0; i < n; i++){
        result = result* p;
    }
    return result;
}

// We use the newton method here.
double x_to_the_power_of_z(double *pDouble, double n) {
    if(n>= 1){
        return x_to_the_power_of_n(pDouble, (int)n)* x_to_the_power_of_z(pDouble, n- ((int)n));
    }
    double xn = n;
    double kehrwertofn = 1/n;
    double xn1 =n;
    double fx, dfx;
    for(int i = 0; i< 1000; i++){
        xn = xn1;
        xn1 = xn- (x_to_the_power_of_n(&xn,kehrwertofn+1)-*pDouble * xn)/(*pDouble * kehrwertofn);
    }
    return xn1;
}

